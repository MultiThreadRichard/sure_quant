// w4a16_ops.cu — SureQuant W4A16 custom operators for RTX 4090 (SM89 / Ada).
//
// Two kernels:
//   1) w4a16_dequant_gemm : int4 x fp16 GEMM with on-the-fly dequantization.
//        z[m,n] = sum_k x[m,k] * ( code[k,n] * scale[k, n>>7] )
//        weight code is signed two's-complement int4, packed 2 values / uint8
//        (low nibble = even column, high nibble = odd column), logical layout [K, N].
//        scale is fp32 [K, N/128], folded onto the weight side during the
//        int4 -> fp16 conversion in shared memory.
//   2) inverse_rotate      : apply R^{-1} = Givens^{-1} . Hadamard^{-1} per
//        128-wide output block (reproduces CompositeBlockRotation.inverse for
//        the "hadamard_givens" order with the full butterfly topology).
//
// fp16 inputs / fp32 accumulation (Tensor Core for the GEMM).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>

using namespace nvcuda;

// ---------------------------------------------------------------------------
// GEMM tile configuration
// ---------------------------------------------------------------------------
#define BM 128          // block tile: rows of output (M dimension)
#define BN 128          // block tile: cols of output (N dimension) == one scale block
#define BK 32           // block tile: reduction (K dimension)
#define NWARP_M 4       // warps along M
#define NWARP_N 2       // warps along N
#define NTHREADS (NWARP_M * NWARP_N * 32)   // 256
#define WM (BM / NWARP_M)   // 32 : warp tile rows
#define WN (BN / NWARP_N)   // 64 : warp tile cols
#define NFRAG_M (WM / 16)   // 2
#define NFRAG_N (WN / 16)   // 4
#define NKSTEP (BK / 16)    // 2

// ---------------------------------------------------------------------------
// device helpers
// ---------------------------------------------------------------------------

// signed int4 nibble [0,15] -> int8 [-8,7]
__device__ __forceinline__ int8_t nibble_to_i8(uint32_t nib) {
    return (int8_t)((nib >= 8u) ? (int)nib - 16 : (int)nib);
}

__device__ __forceinline__ __half i8_to_half(int8_t v) {
    return __float2half((float)v);
}

// cp.async 16-byte copy (global -> shared), SM80+
__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gmem_src) {
    uint32_t saddr = (uint32_t)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gmem_src));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

// ---------------------------------------------------------------------------
// Kernel 1: int4 x fp16 dequant GEMM
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(NTHREADS, 2)
w4a16_dequant_gemm_kernel(
    const __half* __restrict__ A,         // [M, K] fp16
    const uint8_t* __restrict__ B_packed, // [K, N/2] packed int4
    const float*  __restrict__ scale,     // [K, N/128] fp32
    __half* __restrict__ C,               // [M, N] fp16 (pre-rotation output z)
    int M, int N, int K)
{
    __shared__ __half As[2][BM][BK];       // activations (raw fp16), double buffered
    __shared__ __half Bs[2][BK][BN];       // dequantized weights (fp16), double buffered
    __shared__ float frag_buf[NTHREADS / 32][16][16];  // epilogue fp32 staging

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / NWARP_N;    // 0..3
    const int wn   = warp % NWARP_N;    // 0..1

    const int m0     = blockIdx.x * BM;
    const int n0     = blockIdx.y * BN;
    const int sblock = n0 >> 7;         // scale block index (BN == 128 == one block)

    // ---- accumulators ----
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[NFRAG_M][NFRAG_N];
    #pragma unroll
    for (int i = 0; i < NFRAG_M; ++i)
      #pragma unroll
      for (int j = 0; j < NFRAG_N; ++j)
        wmma::fill_fragment(acc[i][j], 0.0f);

    // ---- A load: 16-byte cp.async (8 halfs per copy), zero-fill out-of-bounds ----
    auto load_A_tile = [&](int buf, int kbase) {
        const int nchunk = (BM * BK / 8) / NTHREADS;   // 512 / 256 = 2
        #pragma unroll
        for (int t = 0; t < nchunk; ++t) {
            int idx = tid + t * NTHREADS;               // 0 .. BM*BK/8 - 1
            int m   = idx / (BK / 8);                   // row (0..127)
            int k8  = (idx % (BK / 8)) * 8;             // k offset (0,8,16,24)
            int gm  = m0 + m;
            int gk  = kbase + k8;
            __half* sdst = &As[buf][m][k8];
            if (gm < M && gk + 8 <= K) {
                const __half* gsrc = A + (size_t)gm * K + gk;
                cp_async16(sdst, gsrc);
            } else {
                #pragma unroll
                for (int j = 0; j < 8; ++j) sdst[j] = __float2half(0.0f);
            }
        }
    };

    // ---- B load: synchronous int4 -> fp16 dequant, scale folded onto weight ----
    auto load_B_tile = [&](int buf, int kbase) {
        const int nbyte = (BK * BN / 2) / NTHREADS;     // 32*64 / 256 = 8
        #pragma unroll
        for (int t = 0; t < nbyte; ++t) {
            int idx = tid + t * NTHREADS;               // 0 .. BK*BN/2 - 1
            int k   = idx / (BN / 2);                   // k in K-tile (0..31)
            int n2  = idx % (BN / 2);                   // byte index within row (0..63)
            int gk  = kbase + k;
            int gn  = n0 + n2 * 2;
            uint8_t byte = 0;
            if (gk < K && gn + 1 < N)
                byte = B_packed[(size_t)gk * (N / 2) + (n0 >> 1) + n2];
            int8_t c0 = nibble_to_i8(byte & 0x0F);
            int8_t c1 = nibble_to_i8(byte >> 4);
            __half sc = __float2half(__ldg(&scale[(size_t)gk * (N >> 7) + sblock]));
            Bs[buf][k][n2 * 2]     = __hmul(i8_to_half(c0), sc);
            Bs[buf][k][n2 * 2 + 1] = __hmul(i8_to_half(c1), sc);
        }
    };

    // ---- prologue: load K-tile 0 ----
    load_A_tile(0, 0);
    cp_async_commit();
    load_B_tile(0, 0);
    cp_async_wait<0>();
    __syncthreads();

    const int ntiles = K / BK;
    for (int kt = 0; kt < ntiles; ++kt) {
        const int cur = kt & 1;
        const int nxt = (kt + 1) & 1;

        if (kt + 1 < ntiles) {
            load_A_tile(nxt, (kt + 1) * BK);
            cp_async_commit();
            load_B_tile(nxt, (kt + 1) * BK);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < NKSTEP; ++ks) {
            int kbase = ks * 16;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[NFRAG_M];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb[NFRAG_N];
            #pragma unroll
            for (int i = 0; i < NFRAG_M; ++i)
                wmma::load_matrix_sync(fa[i], &As[cur][wm * WM + i * 16][kbase], BK);
            #pragma unroll
            for (int j = 0; j < NFRAG_N; ++j)
                wmma::load_matrix_sync(fb[j], &Bs[cur][kbase][wn * WN + j * 16], BN);
            #pragma unroll
            for (int i = 0; i < NFRAG_M; ++i)
              #pragma unroll
              for (int j = 0; j < NFRAG_N; ++j)
                wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        __syncthreads();
    }

    // ---- epilogue: store fp32 accum -> shared -> fp16 global C ----
    // (store_matrix_sync cannot target local/stack memory; shared is fine.)
    #pragma unroll
    for (int i = 0; i < NFRAG_M; ++i)
      #pragma unroll
      for (int j = 0; j < NFRAG_N; ++j) {
        int cm = m0 + wm * WM + i * 16;
        int cn = n0 + wn * WN + j * 16;
        wmma::store_matrix_sync(&frag_buf[warp][0][0], acc[i][j], 16, wmma::mem_row_major);
        // frag_buf[warp] now holds the 16x16 fragment row-major; convert + write.
        #pragma unroll
        for (int e = lane; e < 256; e += 32) {
            int r = e >> 4, c = e & 15;
            if (cm + r < M && cn + c < N)
                C[(size_t)(cm + r) * N + (cn + c)] = __float2half(frag_buf[warp][r][c]);
        }
      }
}

// ---------------------------------------------------------------------------
// Kernel 1b: v2 int4 x fp16 dequant GEMM — BK=64, cp.async double-buffered,
//            B staged packed (cp.async) then dequantized smem->smem overlapped
//            with the mma of the previous K-tile.  Targets compute-bound prefill.
// ---------------------------------------------------------------------------
#define BM2 128
#define BN2 128
#define BK2 64
#define NWARP_M2 4
#define NWARP_N2 2
#define NTHREADS2 (NWARP_M2 * NWARP_N2 * 32)   // 256
#define WM2 (BM2 / NWARP_M2)   // 32
#define WN2 (BN2 / NWARP_N2)   // 64
#define NFRAG_M2 (WM2 / 16)    // 2
#define NFRAG_N2 (WN2 / 16)    // 4
#define NKSTEP2 (BK2 / 16)     // 4

__global__ void __launch_bounds__(NTHREADS2, 1)
w4a16_dequant_gemm_v2_kernel(
    const __half* __restrict__ A,          // [M, K] fp16
    const uint8_t* __restrict__ B_packed,  // [K, N/2] packed int4
    const float*  __restrict__ scale,      // [K, N/128] fp32
    __half* __restrict__ C,                // [M, N] fp16 (pre-rotation z)
    int M, int N, int K)
{
    // Dynamic shared memory (opt-in > 48KB):  As | Bsp | Bs | frag_buf
    extern __shared__ char smem[];
    const int AS_BUF  = BM2 * BK2;            // 8192 halfs
    const int BSP_BUF = BK2 * (BN2 / 2);      // 4096 bytes
    const int BS_BUF  = BK2 * BN2;            // 8192 halfs
    __half*   As      = reinterpret_cast<__half*>(smem);
    uint8_t*  Bsp     = reinterpret_cast<uint8_t*>(As + 2 * AS_BUF);
    __half*   Bs      = reinterpret_cast<__half*>(Bsp + 2 * BSP_BUF);
    float*    frag_buf = reinterpret_cast<float*>(Bs + 2 * BS_BUF);

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int wm     = warp / NWARP_N2;
    const int wn     = warp % NWARP_N2;
    const int m0     = blockIdx.x * BM2;
    const int n0     = blockIdx.y * BN2;
    const int sblock = n0 >> 7;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[NFRAG_M2][NFRAG_N2];
    #pragma unroll
    for (int i = 0; i < NFRAG_M2; ++i)
      #pragma unroll
      for (int j = 0; j < NFRAG_N2; ++j)
        wmma::fill_fragment(acc[i][j], 0.0f);

    // A tile: 16-byte cp.async (8 halfs each), zero-fill OOB.
    auto load_A_async = [&](int buf, int kbase) {
        const int nchunk = (BM2 * BK2 / 8) / NTHREADS2;   // 128*64/8/256 = 4
        #pragma unroll
        for (int t = 0; t < nchunk; ++t) {
            int idx = tid + t * NTHREADS2;                // 0 .. 1023
            int m   = idx / (BK2 / 8);                    // row 0..127
            int k8  = (idx % (BK2 / 8)) * 8;              // k offset 0,8,..56
            int gm  = m0 + m;
            int gk  = kbase + k8;
            __half* sdst = &As[buf * AS_BUF + m * BK2 + k8];
            if (gm < M && gk + 8 <= K)
                cp_async16(sdst, A + (size_t)gm * K + gk);
            else
                #pragma unroll
                for (int j = 0; j < 8; ++j) sdst[j] = __float2half(0.0f);
        }
    };

    // B tile (packed): 256 chunks of 16B == 64 rows x 4 chunks/row.
    auto load_B_async = [&](int buf, int kbase) {
        int c    = tid;                    // 0..255
        int k    = c >> 2;                 // row 0..63
        int coff = (c & 3) * 16;           // byte offset 0,16,32,48
        int gk   = kbase + k;
        if (gk < K && n0 + BN2 <= N)
            cp_async16(&Bsp[buf * BSP_BUF + k * (BN2 / 2) + coff],
                       B_packed + (size_t)gk * (N / 2) + (n0 >> 1) + coff);
        else
            #pragma unroll
            for (int j = 0; j < 16; ++j) Bsp[buf * BSP_BUF + k * (BN2 / 2) + coff + j] = 0;
    };

    // Dequant packed smem -> fp16 smem, folding the (k-dependent) scale.
    auto dequant_B = [&](int buf, int kbase) {
        const int nbyte = (BK2 * BN2 / 2) / NTHREADS2;     // 64*64/256 = 16
        #pragma unroll
        for (int t = 0; t < nbyte; ++t) {
            int idx = tid + t * NTHREADS2;                 // 0 .. 4095
            int k   = idx >> 6;                            // row 0..63
            int n2  = idx & 63;                            // byte within row
            int gk  = kbase + k;
            uint8_t byte = Bsp[buf * BSP_BUF + k * (BN2 / 2) + n2];
            int8_t c0 = nibble_to_i8(byte & 0x0F);
            int8_t c1 = nibble_to_i8(byte >> 4);
            __half sc = (gk < K)
                ? __float2half(__ldg(&scale[(size_t)gk * (N >> 7) + sblock]))
                : __float2half(0.0f);
            Bs[buf * BS_BUF + k * BN2 + n2 * 2]     = __hmul(i8_to_half(c0), sc);
            Bs[buf * BS_BUF + k * BN2 + n2 * 2 + 1] = __hmul(i8_to_half(c1), sc);
        }
    };

    // ---- prologue: stage K-tile 0 ----
    load_A_async(0, 0);
    load_B_async(0, 0);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();          // make all threads' cp.async data visible before dequant
    dequant_B(0, 0);
    __syncthreads();

    const int ntiles = (K + BK2 - 1) / BK2;
    for (int kt = 0; kt < ntiles; ++kt) {
        const int cur = kt & 1;
        // prefetch next tile (async) into the other buffer
        if (kt + 1 < ntiles) {
            load_A_async(cur ^ 1, (kt + 1) * BK2);
            load_B_async(cur ^ 1, (kt + 1) * BK2);
            cp_async_commit();
        }
        // mma on current tile
        #pragma unroll
        for (int ks = 0; ks < NKSTEP2; ++ks) {
            int kbase = ks * 16;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[NFRAG_M2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb[NFRAG_N2];
            #pragma unroll
            for (int i = 0; i < NFRAG_M2; ++i)
                wmma::load_matrix_sync(fa[i], &As[cur * AS_BUF + (wm * WM2 + i * 16) * BK2 + kbase], BK2);
            #pragma unroll
            for (int j = 0; j < NFRAG_N2; ++j)
                wmma::load_matrix_sync(fb[j], &Bs[cur * BS_BUF + kbase * BN2 + wn * WN2 + j * 16], BN2);
            #pragma unroll
            for (int i = 0; i < NFRAG_M2; ++i)
              #pragma unroll
              for (int j = 0; j < NFRAG_N2; ++j)
                wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        // finish staging the prefetched tile (overlapped with mma above)
        if (kt + 1 < ntiles) {
            cp_async_wait<0>();
            __syncthreads();        // cp.async data (Bsp/As next) visible to all threads
            dequant_B(cur ^ 1, (kt + 1) * BK2);
        }
        __syncthreads();
    }

    // ---- epilogue: fp32 accum -> shared -> fp16 global C ----
    #pragma unroll
    for (int i = 0; i < NFRAG_M2; ++i)
      #pragma unroll
      for (int j = 0; j < NFRAG_N2; ++j) {
        int cm = m0 + wm * WM2 + i * 16;
        int cn = n0 + wn * WN2 + j * 16;
        wmma::store_matrix_sync(&frag_buf[warp * 256], acc[i][j], 16, wmma::mem_row_major);
        #pragma unroll
        for (int e = lane; e < 256; e += 32) {
            int r = e >> 4, c = e & 15;
            if (cm + r < M && cn + c < N)
                C[(size_t)(cm + r) * N + (cn + c)] = __float2half(frag_buf[warp * 256 + r * 16 + c]);
        }
      }
}

// ---------------------------------------------------------------------------
// Kernel 1c: v3 int4 x fp16 dequant GEMM — BK=32 (2 blocks/SM) but with the same
//            cp.async double-buffered, dequant-overlapped pipeline as v2.  Keeps
//            the higher occupancy of v1 while removing the synchronous B load.
// ---------------------------------------------------------------------------
#define BM3 128
#define BN3 128
#define BK3 32
#define NWARP_M3 4
#define NWARP_N3 2
#define NTHREADS3 (NWARP_M3 * NWARP_N3 * 32)   // 256
#define WM3 (BM3 / NWARP_M3)   // 32
#define WN3 (BN3 / NWARP_N3)   // 64
#define NFRAG_M3 (WM3 / 16)    // 2
#define NFRAG_N3 (WN3 / 16)    // 4
#define NKSTEP3 (BK3 / 16)     // 2

__global__ void __launch_bounds__(NTHREADS3, 2)
w4a16_dequant_gemm_v3_kernel(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B_packed,
    const float*  __restrict__ scale,
    __half* __restrict__ C,
    int M, int N, int K)
{
    extern __shared__ char smem[];
    const int AS_BUF  = BM3 * BK3;            // 4096 halfs
    const int BSP_BUF = BK3 * (BN3 / 2);      // 2048 bytes
    const int BS_BUF  = BK3 * BN3;            // 4096 halfs
    __half*  As      = reinterpret_cast<__half*>(smem);
    uint8_t* Bsp     = reinterpret_cast<uint8_t*>(As + 2 * AS_BUF);
    __half*  Bs      = reinterpret_cast<__half*>(Bsp + 2 * BSP_BUF);
    float*   frag_buf = reinterpret_cast<float*>(Bs + 2 * BS_BUF);

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int wm     = warp / NWARP_N3;
    const int wn     = warp % NWARP_N3;
    const int m0     = blockIdx.x * BM3;
    const int n0     = blockIdx.y * BN3;
    const int sblock = n0 >> 7;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[NFRAG_M3][NFRAG_N3];
    #pragma unroll
    for (int i = 0; i < NFRAG_M3; ++i)
      #pragma unroll
      for (int j = 0; j < NFRAG_N3; ++j)
        wmma::fill_fragment(acc[i][j], 0.0f);

    auto load_A_async = [&](int buf, int kbase) {
        const int nchunk = (BM3 * BK3 / 8) / NTHREADS3;   // 128*32/8/256 = 2
        #pragma unroll
        for (int t = 0; t < nchunk; ++t) {
            int idx = tid + t * NTHREADS3;
            int m   = idx / (BK3 / 8);
            int k8  = (idx % (BK3 / 8)) * 8;
            int gm  = m0 + m;
            int gk  = kbase + k8;
            __half* sdst = &As[buf * AS_BUF + m * BK3 + k8];
            if (gm < M && gk + 8 <= K)
                cp_async16(sdst, A + (size_t)gm * K + gk);
            else
                #pragma unroll
                for (int j = 0; j < 8; ++j) sdst[j] = __float2half(0.0f);
        }
    };

    // 32 rows x 4 chunks/row = 128 chunks; threads 0..127 participate.
    auto load_B_async = [&](int buf, int kbase) {
        if (tid < BK3 * 4) {
            int k    = tid >> 2;
            int coff = (tid & 3) * 16;
            int gk   = kbase + k;
            if (gk < K && n0 + BN3 <= N)
                cp_async16(&Bsp[buf * BSP_BUF + k * (BN3 / 2) + coff],
                           B_packed + (size_t)gk * (N / 2) + (n0 >> 1) + coff);
            else
                #pragma unroll
                for (int j = 0; j < 16; ++j) Bsp[buf * BSP_BUF + k * (BN3 / 2) + coff + j] = 0;
        }
    };

    auto dequant_B = [&](int buf, int kbase) {
        const int nbyte = (BK3 * BN3 / 2) / NTHREADS3;     // 32*64/256 = 8
        #pragma unroll
        for (int t = 0; t < nbyte; ++t) {
            int idx = tid + t * NTHREADS3;
            int k   = idx >> 6;
            int n2  = idx & 63;
            int gk  = kbase + k;
            uint8_t byte = Bsp[buf * BSP_BUF + k * (BN3 / 2) + n2];
            int8_t c0 = nibble_to_i8(byte & 0x0F);
            int8_t c1 = nibble_to_i8(byte >> 4);
            __half sc = (gk < K)
                ? __float2half(__ldg(&scale[(size_t)gk * (N >> 7) + sblock]))
                : __float2half(0.0f);
            Bs[buf * BS_BUF + k * BN3 + n2 * 2]     = __hmul(i8_to_half(c0), sc);
            Bs[buf * BS_BUF + k * BN3 + n2 * 2 + 1] = __hmul(i8_to_half(c1), sc);
        }
    };

    load_A_async(0, 0);
    load_B_async(0, 0);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    dequant_B(0, 0);
    __syncthreads();

    const int ntiles = (K + BK3 - 1) / BK3;
    for (int kt = 0; kt < ntiles; ++kt) {
        const int cur = kt & 1;
        if (kt + 1 < ntiles) {
            load_A_async(cur ^ 1, (kt + 1) * BK3);
            load_B_async(cur ^ 1, (kt + 1) * BK3);
            cp_async_commit();
        }
        #pragma unroll
        for (int ks = 0; ks < NKSTEP3; ++ks) {
            int kbase = ks * 16;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[NFRAG_M3];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb[NFRAG_N3];
            #pragma unroll
            for (int i = 0; i < NFRAG_M3; ++i)
                wmma::load_matrix_sync(fa[i], &As[cur * AS_BUF + (wm * WM3 + i * 16) * BK3 + kbase], BK3);
            #pragma unroll
            for (int j = 0; j < NFRAG_N3; ++j)
                wmma::load_matrix_sync(fb[j], &Bs[cur * BS_BUF + kbase * BN3 + wn * WN3 + j * 16], BN3);
            #pragma unroll
            for (int i = 0; i < NFRAG_M3; ++i)
              #pragma unroll
              for (int j = 0; j < NFRAG_N3; ++j)
                wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        if (kt + 1 < ntiles) {
            cp_async_wait<0>();
            __syncthreads();
            dequant_B(cur ^ 1, (kt + 1) * BK3);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < NFRAG_M3; ++i)
      #pragma unroll
      for (int j = 0; j < NFRAG_N3; ++j) {
        int cm = m0 + wm * WM3 + i * 16;
        int cn = n0 + wn * WN3 + j * 16;
        wmma::store_matrix_sync(&frag_buf[warp * 256], acc[i][j], 16, wmma::mem_row_major);
        #pragma unroll
        for (int e = lane; e < 256; e += 32) {
            int r = e >> 4, c = e & 15;
            if (cm + r < M && cn + c < N)
                C[(size_t)(cm + r) * N + (cn + c)] = __float2half(frag_buf[warp * 256 + r * 16 + c]);
        }
      }
}

// ---------------------------------------------------------------------------
// Kernel 2: int4 x fp16 GEMV (decode, M == 1) — bandwidth-bound, reads 4x less
// ---------------------------------------------------------------------------
// Split-K: grid = (N/128, K/BK_GEMV) blocks, 128 threads each.  Each block
// reduces one K-chunk of one 128-column scale block; partial sums are combined
// with float atomics.  High occupancy keeps the strided column read coalesced.
#define BK_GEMV 128

__global__ void w4a16_gemv_kernel(
    const __half* __restrict__ x,        // [K] (single token, M==1)
    const uint8_t* __restrict__ B_packed,// [K, N/2] packed int4
    const float*  __restrict__ scale,    // [K, N/128] fp32
    float* __restrict__ y_f32,           // [N] fp32 accumulator (zero-initialized)
    int N, int K)
{
    const int tid    = threadIdx.x;
    const int sblock = blockIdx.x;       // which 128-column scale block
    const int kc     = blockIdx.y;       // which K chunk
    const int n      = sblock * 128 + tid;
    if (n >= N) return;

    const int nb  = n >> 1;
    const int nib = n & 1;
    const int kstart = kc * BK_GEMV;
    const int kend   = min(kstart + BK_GEMV, K);

    float acc = 0.0f;
    for (int k = kstart; k < kend; ++k) {
        uint8_t byte = B_packed[(size_t)k * (N / 2) + nb];
        int c = nibble_to_i8(nib ? (byte >> 4) : (byte & 0x0F));
        float s = __ldg(&scale[(size_t)k * (N >> 7) + sblock]);
        acc += __half2float(x[k]) * ((float)c * s);
    }
    atomicAdd(&y_f32[n], acc);
}

// ---------------------------------------------------------------------------
// Kernel 3: inverse rotation epilogue (Givens^{-1} then Hadamard^{-1})
// ---------------------------------------------------------------------------
// Reproduces CompositeBlockRotation.inverse("hadamard_givens") per 128-block:
//   givens.inverse  : full butterfly pairs applied stride 64 -> 1, angle = -theta
//   hadamard.inverse: FWHT (normalised by 1/sqrt(128)) then element-wise x signs
// One thread block per (row, 128-block); 128 threads.
__global__ void inverse_rotate_kernel(
    const __half* __restrict__ z,       // [M, N] fp16
    __half* __restrict__ y,             // [M, N] fp16
    int M, int N,
    const float* __restrict__ signs,    // [N/128, 128]  (+-1)
    const float* __restrict__ theta_cos,// [N/128, 448]  cos(-theta) == cos(theta)
    const float* __restrict__ theta_sin)// [N/128, 448]  sin(-theta) == -sin(theta)
{
    __shared__ float s[128];
    const int b   = blockIdx.y;          // which 128-block
    const int m   = blockIdx.x;          // which row/token
    const int col = threadIdx.x;         // 0..127

    const int gn = b * 128 + col;
    if (m < M && gn < N)
        s[col] = __half2float(z[(size_t)m * N + gn]);
    else
        s[col] = 0.0f;
    __syncthreads();

    // ---- 1) Givens inverse: butterfly, stride 64 -> 1, negated angle ----
    const float* tc = &theta_cos[(size_t)b * 448];
    const float* ts = &theta_sin[(size_t)b * 448];
    #pragma unroll
    for (int L = 6; L >= 0; --L) {
        int stride = 1 << L;
        int base   = L * 64;             // theta base index for this stride
        if (col < 64) {                  // 64 disjoint pairs per stride
            int j     = col;             // pair index within stride
            int start = (j / stride) * (2 * stride);
            int i     = j % stride;
            int p     = start + i;
            int q     = p + stride;
            int k     = base + j;
            float c = tc[k], sn = ts[k];
            float xp = s[p], xq = s[q];
            s[p] = c * xp - sn * xq;
            s[q] = sn * xp + c * xq;
        }
        __syncthreads();
    }

    // ---- 2) Hadamard inverse: FWHT then x signs ----
    #pragma unroll
    for (int h = 1; h < 128; h <<= 1) {
        if ((col & h) == 0) {
            int partner = col + h;
            float a = s[col], b = s[partner];
            s[col]    = a + b;
            s[partner] = a - b;
        }
        __syncthreads();
    }
    s[col] = s[col] * (0.08838834764831845f) * signs[(size_t)b * 128 + col];

    if (m < M && gn < N)
        y[(size_t)m * N + gn] = __float2half(s[col]);
}

// ---------------------------------------------------------------------------
// host launchers + torch bindings
// ---------------------------------------------------------------------------

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor w4a16_gemm(torch::Tensor A, torch::Tensor B_packed, torch::Tensor scale) {
    CHECK_CUDA(A); CHECK_CUDA(B_packed); CHECK_CUDA(scale);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B_packed); CHECK_CONTIGUOUS(scale);

    TORCH_CHECK(A.dim() == 2, "A must be [M, K]");
    TORCH_CHECK(B_packed.dim() == 2, "B_packed must be [K, N/2]");
    TORCH_CHECK(scale.dim() == 2, "scale must be [K, N/128]");
    TORCH_CHECK(A.scalar_type() == at::kHalf, "A must be fp16");
    TORCH_CHECK(B_packed.scalar_type() == at::kByte, "B_packed must be uint8");
    TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale must be fp32");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B_packed.size(1) * 2;
    TORCH_CHECK(B_packed.size(0) == K, "B_packed rows must equal K");
    TORCH_CHECK(scale.size(0) == K, "scale rows must equal K");
    TORCH_CHECK(scale.size(1) == N / 128, "scale cols must equal N/128");

    auto C = torch::empty({M, N}, A.options());

    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    w4a16_dequant_gemm_kernel<<<grid, NTHREADS, 0, stream>>>(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        B_packed.data_ptr<uint8_t>(),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K);
    return C;
}

torch::Tensor w4a16_gemv(torch::Tensor x, torch::Tensor B_packed, torch::Tensor scale) {
    CHECK_CUDA(x); CHECK_CUDA(B_packed); CHECK_CUDA(scale);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(B_packed); CHECK_CONTIGUOUS(scale);

    TORCH_CHECK(x.dim() == 1, "x must be [K] (single token)");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
    TORCH_CHECK(B_packed.dim() == 2 && B_packed.scalar_type() == at::kByte, "B_packed must be [K, N/2] uint8");
    TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale must be fp32");

    const int K = x.size(0);
    const int N = B_packed.size(1) * 2;
    TORCH_CHECK(B_packed.size(0) == K, "B_packed rows must equal K");
    TORCH_CHECK(scale.size(0) == K && scale.size(1) == N / 128, "scale must be [K, N/128]");

    auto y_f32 = torch::zeros({N}, x.options().dtype(at::kFloat));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(N / 128, (K + BK_GEMV - 1) / BK_GEMV);
    w4a16_gemv_kernel<<<grid, 128, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        B_packed.data_ptr<uint8_t>(),
        scale.data_ptr<float>(),
        y_f32.data_ptr<float>(),
        N, K);
    return y_f32;
}

torch::Tensor inverse_rotate(torch::Tensor z, torch::Tensor signs,
                             torch::Tensor theta_cos, torch::Tensor theta_sin) {
    CHECK_CUDA(z); CHECK_CUDA(signs); CHECK_CUDA(theta_cos); CHECK_CUDA(theta_sin);
    CHECK_CONTIGUOUS(z); CHECK_CONTIGUOUS(signs);
    CHECK_CONTIGUOUS(theta_cos); CHECK_CONTIGUOUS(theta_sin);

    TORCH_CHECK(z.dim() == 2, "z must be [M, N]");
    TORCH_CHECK(z.scalar_type() == at::kHalf, "z must be fp16");

    const int M = z.size(0);
    const int N = z.size(1);
    const int nb = N / 128;
    TORCH_CHECK(N % 128 == 0, "N must be divisible by 128");
    TORCH_CHECK(signs.size(0) == nb && signs.size(1) == 128, "signs must be [N/128, 128]");
    TORCH_CHECK(theta_cos.size(0) == nb && theta_cos.size(1) == 448, "theta_cos must be [N/128, 448]");
    TORCH_CHECK(theta_sin.size(0) == nb && theta_sin.size(1) == 448, "theta_sin must be [N/128, 448]");

    auto y = torch::empty({M, N}, z.options());
    dim3 grid(M, nb);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    inverse_rotate_kernel<<<grid, 128, 0, stream>>>(
        reinterpret_cast<const __half*>(z.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        M, N,
        signs.data_ptr<float>(),
        theta_cos.data_ptr<float>(),
        theta_sin.data_ptr<float>());
    return y;
}

torch::Tensor w4a16_gemm_v2(torch::Tensor A, torch::Tensor B_packed, torch::Tensor scale) {
    CHECK_CUDA(A); CHECK_CUDA(B_packed); CHECK_CUDA(scale);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B_packed); CHECK_CONTIGUOUS(scale);

    TORCH_CHECK(A.dim() == 2, "A must be [M, K]");
    TORCH_CHECK(B_packed.dim() == 2, "B_packed must be [K, N/2]");
    TORCH_CHECK(scale.dim() == 2, "scale must be [K, N/128]");
    TORCH_CHECK(A.scalar_type() == at::kHalf, "A must be fp16");
    TORCH_CHECK(B_packed.scalar_type() == at::kByte, "B_packed must be uint8");
    TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale must be fp32");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B_packed.size(1) * 2;
    TORCH_CHECK(B_packed.size(0) == K, "B_packed rows must equal K");
    TORCH_CHECK(scale.size(0) == K, "scale rows must equal K");
    TORCH_CHECK(scale.size(1) == N / 128, "scale cols must equal N/128");

    auto C = torch::empty({M, N}, A.options());

    // opt-in shared memory (As+Bsp+Bs+frag ~80KB > 48KB default)
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(w4a16_dequant_gemm_v2_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        attr_set = true;
    }

    dim3 grid((M + BM2 - 1) / BM2, (N + BN2 - 1) / BN2);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int smem = (2 * BM2 * BK2 * 2) + (2 * BK2 * (BN2 / 2)) + (2 * BK2 * BN2 * 2) + (8 * 16 * 16 * 4);
    w4a16_dequant_gemm_v2_kernel<<<grid, NTHREADS2, smem, stream>>>(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        B_packed.data_ptr<uint8_t>(),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K);
    return C;
}

torch::Tensor w4a16_gemm_v3(torch::Tensor A, torch::Tensor B_packed, torch::Tensor scale) {
    CHECK_CUDA(A); CHECK_CUDA(B_packed); CHECK_CUDA(scale);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B_packed); CHECK_CONTIGUOUS(scale);

    TORCH_CHECK(A.dim() == 2, "A must be [M, K]");
    TORCH_CHECK(B_packed.dim() == 2, "B_packed must be [K, N/2]");
    TORCH_CHECK(scale.dim() == 2, "scale must be [K, N/128]");
    TORCH_CHECK(A.scalar_type() == at::kHalf, "A must be fp16");
    TORCH_CHECK(B_packed.scalar_type() == at::kByte, "B_packed must be uint8");
    TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale must be fp32");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B_packed.size(1) * 2;
    TORCH_CHECK(B_packed.size(0) == K, "B_packed rows must equal K");
    TORCH_CHECK(scale.size(0) == K, "scale rows must equal K");
    TORCH_CHECK(scale.size(1) == N / 128, "scale cols must equal N/128");

    auto C = torch::empty({M, N}, A.options());

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(w4a16_dequant_gemm_v3_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        attr_set = true;
    }

    dim3 grid((M + BM3 - 1) / BM3, (N + BN3 - 1) / BN3);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int smem = (2 * BM3 * BK3 * 2) + (2 * BK3 * (BN3 / 2)) + (2 * BK3 * BN3 * 2) + (8 * 16 * 16 * 4);
    w4a16_dequant_gemm_v3_kernel<<<grid, NTHREADS3, smem, stream>>>(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        B_packed.data_ptr<uint8_t>(),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w4a16_gemm", &w4a16_gemm, "int4 x fp16 dequant GEMM (scale folded to weight)");
    m.def("w4a16_gemm_v2", &w4a16_gemm_v2, "v2 int4 x fp16 dequant GEMM (BK=64 pipelined)");
    m.def("w4a16_gemm_v3", &w4a16_gemm_v3, "v3 int4 x fp16 dequant GEMM (BK=32 pipelined, 2 blocks/SM)");
    m.def("w4a16_gemv", &w4a16_gemv, "int4 x fp16 GEMV (decode, M==1)");
    m.def("inverse_rotate", &inverse_rotate, "inverse rotation epilogue (Givens^-1 then Hadamard^-1)");
}

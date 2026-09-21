# W4A16 自定义算子设计方案与实测报告（SureQuant / LLaVA-1.5-7B）

> 目标硬件：RTX 4090（SM89 / Ada），可复现命令见文末附录。
> 代码：`cuda_op/w4a16_ops.cu`（kernel）+ `w4a16_ops.py`（python 封装）
> 测试：`cuda_op/test_w4a16_ops.py`、`cuda_op/bench_gemm.py`

---

## 0. 执行摘要（TL;DR）

| 结论 | 内容 |
|---|---|
| 交付算子 | 4 个：反量化 GEMM（3 个版本 v1/v2/v3）、反量化 GEMV、逆旋转 epilogue |
| 正确性 | 全部通过。GEMM/GEMV rel ≤ 5.6e-4，ROT rel ≤ 1.1e-3，**均落在 fp16 单位舍入误差 u = 2⁻¹¹ = 4.88e-4 的量级内** |
| Decode 收益 | **int4 GEMV 相对 fp16 cuBLAS 加速 1.86x ~ 2.64x**（大 K 场景），且显存下降 3.76x |
| Prefill 收益 | **int4 GEMM 未能追平 fp16 cuBLAS**，最好 0.69x（v1 / v3）。v2 因占用率塌陷反而更差（0.35x） |
| Prefill 上限根因 | 权重 scale 粒度是 `per_vector_block` → scale 形状 `[K, N/128]`，**与 k 相关**，无法从 K 维归约中提出 → **无法使用 int4/int8 Tensor Core**，只能走 fp16 TC 再叠加反量化开销 |
| 显存收益 | int4 权重（含 scale）3.612 GB vs fp16 13.598 GB，**节省 9.99 GB（3.76x）** |

**一句话**：这套算子的价值在 **decode 提速 + 显存压缩 3.76x**；prefill 在 W4A16 格式下无法超过 cuBLAS fp16，这是格式决定的硬上限，不是实现问题（详见 §6）。

---

## 1. 算子清单与调用契约

| 算子 | 输入 | 输出 | 适用阶段 |
|---|---|---|---|
| `w4a16_gemm` (v1) | A`[M,K]`fp16, B_packed`[K,N/2]`u8, scale`[K,N/128]`fp32 | C`[M,N]`fp16 | prefill（基线） |
| `w4a16_gemm_v2` | 同上 | 同上 | prefill（BK=64 流水线，实验） |
| `w4a16_gemm_v3` | 同上 | 同上 | prefill（BK=32 流水线，实验） |
| `w4a16_gemv` | x`[K]`fp16, B_packed, scale | y`[N]`fp16 | decode（M==1） |
| `inverse_rotate` | z`[M,N]`fp16, signs`[N/128,128]`fp32, θcos/θsin`[N/128,448]`fp32 | y`[M,N]`fp16 | 两个阶段共用的 epilogue |

GEMM 计算式：

```
z[m,n] = Σ_k  x[m,k] · ( code[k,n] · scale[k, n>>7] )
```

---

## 2. 数据格式（与 `persistence.py` 严格对齐）

- **权重 int4**：有符号二进制补码，**2 个值打包进 1 个 uint8**（低半字节 = 偶数列，高半字节 = 奇数列），逻辑布局 `[K, N]`，物理布局 `[K, N/2]` uint8。
- **scale fp32，形状 `[K, N/128]`**：粒度 `per_vector_block`，即每个 (k 行, 128 列块) 一个 scale。**这个形状是本项目所有性能结论的根源**。
- **激活 A**：fp16（部署路径为 W4A16，激活量化器在校准时使用、落盘时剥离）。
- **累加**：fp32。
- 每元素存储开销 = 0.5 B（int4）+ 4/128 B（scale）= **0.5312 B**，对 fp16 的 2 B 为 **3.76x**。

---

## 3. 总体流水线

SureQuant 的推理路径是「旋转域 GEMM + 逆旋转 epilogue」两阶段：

```
x ──► [int4 反量化 GEMM] ──► z = x·W_rot ──► [逆旋转 R⁻¹] ──► y
       权重常驻 int4，反量化在片上完成       Givens⁻¹ ∘ Hadamard⁻¹，按 128 块
```

- **阶段 1（GEMM/GEMV）**：权重以 int4 存于显存，在 shared memory 中即时反量化成 fp16，scale 直接折到权重侧，再交给 fp16 Tensor Core。激活保持 fp16 不做量化。
- **阶段 2（epilogue）**：把 GEMM 输出从旋转域变回原坐标系。它只与输出块内的 128 个元素有关，因此可以按 (行, 128列块) 完全并行。

---

## 4. Kernel 设计要点

### 4.1 Kernel 1 — 反量化 GEMM（v1，主用版本）

**三级 tiling**：block tile 128×128，warp tile 32×64，WMMA fragment 16×16×16。

| 配置项 | 取值 |
|---|---|
| block tile (BM×BN×BK) | 128 × 128 × 32 |
| warp 划分 | 8 warps = 4(沿M) × 2(沿N)，每 warp 32×64 |
| 每 warp fragment 数 | 2×4 = 8 个 `16×16×16` accumulator |
| 线程数 | 256 |
| 每 SM block 数 | 2（`__launch_bounds__(256, 2)`） |
| shared memory | 静态 48 KB 以内 |

**关键设计点**：

1. **A 侧用 `cp.async` 双缓冲**：16 B 粒度异步拷贝 global→shared，K 方向预取下一块，与当前块的 mma 重叠。越界用零填充，因此**不需要在 K/N 边界加分支**。
2. **反量化折 scale 到权重侧**：`Bs = fp16(int4_code) × fp16(scale[k, n>>7])`。这样 epilogue 只需一次 fp32→fp16 转换，无需额外的 scale 乘法 pass。代价是 scale 必须按 k 逐行读取（见 §6）。
3. **BN = 128 = 一个 scale block**：这是一个刻意的选择——block tile 的 N 宽度恰好等于 scale 的块宽度，因此**每个 block 只需要一个 scale 列索引 `sblock = n0>>7`**，内层循环里 scale 的列地址是常量。
4. **epilogue 借道 shared memory**：`store_matrix_sync` 不能写 local/stack memory，故先把 fp32 accumulator 落到 shared 暂存区，再由线程按行主序转 fp16 写回显存。附带做了 M/N 越界保护。

### 4.2 Kernel 1b / 1c — v2 / v3 流水线实验（Stage 3）

v1 的薄弱点是 **B 侧反量化是同步的，且压在临界路径上**（load → dequant 全部完成后才能开始 mma）。v2/v3 的思路是把 B 也做成异步：**先 `cp.async` 把「打包态」的 int4 搬进 shared，再在 shared→shared 上反量化，并让这次反量化与当前 K 块的 mma 重叠**。

| | v2 | v3 |
|---|---|---|
| BK | 64 | 32 |
| shared memory | 动态 ~80 KB | 动态 ~48 KB |
| block/SM | **1** | **2** |
| B 路径 | `cp.async` 打包态 → 反量化 | 同 v2 |

**两种版本的共同结构**：A 与 B_packed 都由 `cp.async` 预取进另一组缓冲；当前块的 mma 执行期间完成下一块的 smem→smem 反量化；靠 `cp.async.wait_group` + `__syncthreads()` 划清 stage 边界。

**踩坑记录（重要）**：v2 初版出现 rel 1.5e-2 ~ 5.4e-2 的严重错误（应为 ~5e-4），且只在 `N=4096 且 K∈{512,1024}` 等特定形状复现。根因是 **`cp.async.wait_group` 是「逐线程」语义**——它只保证*本线程*发出的拷贝完成，不保证*其他线程*拷进 shared 的数据对本线程可见。而 `dequant_B` 会读取由其他线程的 `cp.async` 写入的字节，中间没有屏障。**修复方式：在 `wait_group` 与消费方之间补 `__syncthreads()`**。修复后三个版本逐位一致。v3 从编写起就带这个屏障。

### 4.3 Kernel 2 — 反量化 GEMV（decode，M==1）

Decode 阶段是**纯带宽受限**（算术强度 1~3.76 FLOP/B，远低于 RTX 4090 的机器平衡点 ~164 FLOP/B），因此设计目标只有「少读字节」。

- **Split-K**：grid = `(N/128, K/128)`，每 block 128 线程。每个 block 负责「一个 128 列 scale 块 × 一个 128 长的 K 段」的部分和。
- 每个线程负责一整列（固定 n），沿 K 串行累加：读 1 个 byte（2 个 int4，取自己那一半）、读 1 个 fp32 scale、读 1 个 fp16 激活，做一次融合乘加。
- **fp32 atomicAdd 归并** partial sum（输出张量预先 zero-init）。
- 因为每个线程走一整列，同 warp 内相邻线程访问的是**相邻的 n**，即 byte 地址差 1 → 全局读取完全合并（coalesced）。
- 收益来源：读的是 int4（0.5 B/元素）而不是 fp16（2 B/元素），**DRAM 流量降到 1/4**，所以理想加速比接近 4x；实测 1.86~2.64x 是因为小 K 时 scale 的 fp32 读取（0.031 B/元素）和 atomic 开销占比上升。

### 4.4 Kernel 3 — 逆旋转 epilogue

严格复现 `CompositeBlockRotation.inverse(order="hadamard_givens")`：

- **并行划分**：一个 block 处理 (一行, 一个 128 列块)，128 个线程，数据放 shared（fp32）。
- **Givens⁻¹**：7 级蝶形（stride 64→1），每级 64 个不相交 pair，角度取 `-θ`（实现上是把 sin 取负、cos 不变，在 host 侧预先算好 cos/sin 表传入，kernel 内不做三角函数）。每级后 `__syncthreads()`，因为 pair 会跨线程。
- **Hadamard⁻¹**：7 级 FWHT（`1,2,4,...,64`），归一化因子 `1/√128 = 0.08838834764831845`，最后逐元素乘 `signs`。
- 边界：`M`/`N` 不整齐时读侧补 0、写侧判断，保证越界安全。

**为什么这个 epilogue 要写成自定义 kernel**：torch 参考实现在 7 级蝶形的每一级都做了一次 fp16 舍入（`ops/givens.py` 里的 `index_copy_` 回写 fp16），而本 kernel 全程 fp32 中间量、只在最后舍入一次。实测本 kernel 反而**比 torch 参考更精确**（0.56~0.98 u vs 2.16~2.45 u），这解释了测试中 ROT 阈值（5e-3）比 GEMM 阈值（2e-2）更严格却依然轻松通过的原因。

---

## 5. 实测结果

### 5.1 Part 1 — 算子计算正确性

度量定义：`rel = max|out − ref| / max|ref|`（最大绝对误差 / 参考值动态范围）。

```
  GEMM  M= 128 K=   32 N=  128: rel=4.209e-04  [OK]
  GEMM  M=  16 K= 4096 N= 1024: rel=5.317e-04  [OK]
  GEMM  M= 128 K= 4096 N= 4096: rel=4.267e-04  [OK]
  GEMM  M=   1 K= 4096 N= 4096: rel=3.928e-04  [OK]
  GEMM  M= 256 K=11008 N= 4096: rel=4.097e-04  [OK]
  GEMM  M= 128 K= 1024 N= 4096: rel=5.516e-04  [OK]
  GEMV  K= 4096 N= 4096: rel=3.193e-04  [OK]
  GEMV  K=11008 N= 4096: rel=2.776e-04  [OK]
  ROT   M=   4 N= 1024: rel=9.906e-04  [OK]
  ROT   M=   1 N= 4096: rel=1.096e-03  [OK]
  ROT   M=   8 N= 4096: rel=9.643e-04  [OK]
  -> correctness PASSED (GEMM rel<=5.5e-04, rotate rel<=1.1e-03)
```

**误差量级解读**：fp16 的存储精度是 10 位尾数，单位舍入误差 `u = 2⁻¹¹ = 4.88e-4`。所有 GEMM/GEMV 的 rel 都在 0.8~1.1 u 之间，ROT 在 2 u 左右。**这已经是 fp16 输出格式能做到的极限**——一个「完全正确」的 fp16 算子不应该、也不可能做到零误差，因为结果本身要被舍入到 fp16 才能存下来。所以判据不是「误差为零」，而是「误差落在数值格式的舍入预算内」。

### 5.2 Part 2 — 推理速度对比（自定义 int4 vs fp16 cuBLAS）

#### Decode（M=1）：int4 GEMV vs fp16 cuBLAS

| shape (K,N) | fp16 cuBLAS | int4 gemv | 加速比 |
|---|---|---|---|
| (4096, 4096) | 0.035 ms | 0.019 ms | **1.86x** |
| (4096, 11008) | 0.084 ms | 0.041 ms | **2.06x** |
| (11008, 4096) | 0.100 ms | 0.038 ms | **2.64x** |
| (1024, 4096) | 0.008 ms | 0.016 ms | 0.52x |

> `(1024, 4096)` 是小 K 场景：绝对耗时只有 8~16 μs，kernel launch 与 atomic 归并的固定开销占比过高，split-K 不再划算。模型实际的主干线性层（K=4096/11008）都在 1.86x 以上。

#### Prefill（M>1）：三个 GEMM 版本 vs fp16 cuBLAS

`bench_gemm.py` 实测（200 次迭代，20 次预热）：

| shape (M,K,N) | cuBLAS | v1 | v2 | v3 | v1/c | v2/c | v3/c |
|---|---|---|---|---|---|---|---|
| (8, 4096, 4096) | 0.022 | 0.237 | 0.335 | 0.255 | 0.09x | 0.07x | 0.09x |
| (64, 4096, 4096) | 0.020 | 0.223 | 0.313 | 0.242 | 0.09x | 0.06x | 0.08x |
| (512, 4096, 4096) | 0.111 | 0.226 | 0.315 | 0.252 | 0.49x | 0.35x | 0.44x |
| (2048, 4096, 4096) | 0.462 | 0.672 | 1.255 | 0.704 | **0.69x** | 0.37x | **0.66x** |
| (512, 4096, 11008) | 0.337 | 0.568 | 0.928 | 0.702 | 0.59x | 0.36x | 0.48x |
| (512, 11008, 4096) | 0.297 | 0.609 | 0.841 | 0.670 | 0.49x | 0.35x | 0.44x |

（单位 ms；`(rel2/rel3) ≈ 4e-4`，与 v1 逐位一致）

`test_w4a16_ops.py` 的主测试日志（口径一致）：

| shape (M,K,N) | fp16 cuBLAS | int4 gemm (v1) | 加速比 |
|---|---|---|---|
| (8, 4096, 4096) | 0.021 ms | 0.223 ms | 0.09x |
| (64, 4096, 4096) | 0.020 ms | 0.224 ms | 0.09x |
| (512, 4096, 4096) | 0.111 ms | 0.229 ms | 0.49x |
| (2048, 4096, 4096) | 0.460 ms | 0.677 ms | **0.68x** |
| (512, 4096, 11008) | 0.334 ms | 0.566 ms | 0.59x |
| (512, 11008, 4096) | 0.305 ms | 0.613 ms | 0.50x |

**Stage 3 的结论是负面的，但要分清两种失败**：

- **M ≤ 512 时是「一波」延迟受限，不是吞吐问题**。补测 v1 在各 M 下的耗时（K=N=4096）可以看清这一点：

  | M | grid | block 数 | 耗时 | 实际 TFLOPs |
  |---|---|---|---|---|
  | 8 | 1×32 | 32 | 0.236 ms | 1.1 |
  | 64 | 1×32 | 32 | 0.237 ms | 9.1 |
  | 128 | 1×32 | 32 | 0.238 ms | 18.1 |
  | 256 | 2×32 | 64 | 0.238 ms | 36.0 |
  | 512 | 4×32 | **128** | **0.239 ms** | 71.8 |
  | 1024 | 8×32 | 256 | 0.361 ms | 95.1 |
  | 2048 | 16×32 | 512 | 0.695 ms | 98.9 |
  | 4096 | 32×32 | 1024 | 1.331 ms | **103.2** |

  M 从 8 到 512 耗时**完全持平**（0.236→0.239 ms），而 block 数从 32 涨到 128（已铺满 128 个 SM）——所以**不是 grid 铺不满 SM**。真正的原因是：M ≤ 512 时整个 kernel 只跑**一波**，墙钟时间由单个 block 的**串行 K 循环延迟**决定。每 block 需串行 128 次迭代（K/BK=128），而 v1 每次迭代的 B 侧是**同步全局加载 + 反量化**（只有 A 走了 `cp.async`），128 次的 DRAM 全延迟串起来 ≈ 0.23 ms。超出一波后才转为吞吐受限，M ≥ 1024 起近似线性。
- **因此 0.09x 的含义是「一波固定成本高 12 倍」，不是「算得慢」**。cuBLAS 在 M=8/64 同样是平的（0.020 / 0.021 ms）——它的一波固定成本比我们低约 12 倍，而不是它在算得更快。steady-state 上 v1 的 103 TFLOPs ≈ 62% 峰值并不难看。这也说明：**小 M 场景的正确对标对象是 §4.3 的 GEMV（decode 路径），而不是 GEMM**。**本表中 M=8/64/128/256/512 各行的 0.09x/0.49x 因此不应被读作吞吐比，只有 M ≥ 1024 的行才是真正的算力对标。**
- **M ≥ 512 时是真正的算力对标区**，此时算术强度 > 2048 FLOP/B，进入计算受限。**v1/v3 只能到 0.68x**——即同样的 fp16 Tensor Core 上，多出来的反量化开销让吞吐掉了约 1/3。
- **v2 的 0.35x 是占用率事故**：BK=64 使 shared memory 需求涨到 ~80 KB，每 SM 只能驻留 1 个 block（占用率从 v1 的 2 block/256 线程掉到 1 block/256 线程），延迟无法被其他 block 掩盖。**代价远超它省下的那次同步反量化**。

### 5.3 Part 3 — 权重显存占用（int4 vs fp16）

真实 checkpoint：`model_saved/llava_7b_surequant_w4a16/best_quantized_model/surequant_int4_weights.pt`

| 项 | 数值 |
|---|---|
| 每元素 | int4 0.500 B + scale 0.0312 B = **0.5312 B** vs fp16 **2.0 B** |
| 每元素压缩比 | **3.76x** |
| 量化层数 | 370 |
| 权重总元素 | 6.799 B |
| int4 权重（含 scale） | **3.612 GB** |
| fp16 权重 | **13.598 GB** |
| **显存节省** | **9.99 GB（3.76x）** |

---

## 6. Prefill 为什么追不平 cuBLAS（硬上限分析）

**这不是实现问题，是数据格式问题。** 链条如下：

1. **scale 与 k 相关**。`per_vector_block` 粒度使 scale 形状为 `[K, N/128]`，即 `scale[k, n>>7]` **随归约维 k 变化**。
2. **因此 int4/int8 Tensor Core 不可用**。int4 TC 的语义是 `D = A_q·B_q·(共享 scale)`——scale 必须是「整条 k 归约上的公因子」才能提到求和号外。而这里 `Σ_k x[k]·code[k,n]·scale[k,n>>7]` 中 scale 在求和号内部，无法分解。
3. **于是只能退回 fp16 Tensor Core**。那么与 cuBLAS fp16 用的是**同一块硬件单元、同一个吞吐上限（165 TFLOPS）**，而我们还额外背了反量化（读 int4、查 scale、转 fp16、写 smem）的工作量。
4. **cuBLAS 已经在 91% 峰值附近**（由 §5.2 的 M=2048 数据反推），留给追赶的空间本来就不到 10%。

**结论**：在 W4A16 + `per_vector_block` 格式下，**prefill 无法超过 cuBLAS fp16**。Marlin 风格的极致重写最多把这个 0.68x 推到 ~0.9x 并趋近 cuBLAS，但不会「超过」。

**真正能超过的路径只有一条**：把 scale 改成**与 k 无关**（权重 `per_vector_block` → `per_block`，得 `[N/128]`），同时把激活也量化成 int8/int4（per-token 粒度）——此时 `Σ_k A_q·W_q` 可以整体走 int8/int4 Tensor Core，理论吞吐是 fp16 TC 的 2x（int8）/ 4x（int4），即便叠加反量化开销也有希望到 **1.4x ~ 2x**。代价是必须**重新校准并重新验证模型精度**（激活量化精度损失远大于权重量化）。

---

## 7. 已知限制

1. **prefill 无加速**（见 §6），当前所有 GEMM 版本都慢于 cuBLAS fp16。
2. **N 必须是 128 的倍数**（`scale` 布局与 BN=128 的绑定），不满足则无法调用。
3. **GEMV 只支持 M==1**，batch 化 decode（M>1 但 <8）仍走 GEMM 路径，没有专门的中间档 kernel。
4. **v2/v3 属于实验代码**，建议主路径继续用 v1，二者保留是为了留证 Stage 3 的结论与数据。
5. **inverse_rotate 的 N 必须是 128 的倍数**，且 `theta` 必须按 448 = 7×64 的完整蝶形拓扑排列。
6. 编译期锁定 `TORCH_CUDA_ARCH_LIST=8.9`，在非 Ada 卡上需要重编。

---

## 附录：复现方式

```bash
cd /home/ecnu01/workspace/sure_quant

# 正确性 + 速度 + 显存 主测试（首次运行会 JIT 编译 .cu，约 1~2 分钟）
python cuda_op/test_w4a16_ops.py

# prefill GEMM 三版本微基准（v1 / v2 / v3 vs cuBLAS）
python cuda_op/bench_gemm.py
```

本次结果落盘：

- `logs/cuda-test/test_w4a16_ops_v2.log`
- `logs/cuda-test/bench_gemm.log`

### 关键文件索引

| 文件 | 作用 |
|---|---|
| `cuda_op/w4a16_ops.cu` | 全部 5 个 kernel + host launcher + pybind 绑定 |
| `cuda_op/w4a16_ops.py` | JIT 编译加载 + python 侧 API 封装 |
| `cuda_op/test_w4a16_ops.py` | 正确性 / 速度 / 显存 三部分测试 |
| `cuda_op/bench_gemm.py` | prefill GEMM 微基准（v1/v2/v3 vs cuBLAS） |
| `llava_quant/llava_wa/persistence.py` | int4 打包格式的权威定义（`_pack_signed_int4`） |
| `ops/givens.py`, `ops/hadamard.py` | 逆旋转 kernel 的参考实现（含舍入行为对比依据） |

---

## 附录 B：本设计参考的 RTX 4090（SM89 / Ada）硬件信息

### B.1 本机实测值

采集方式：`torch.cuda.get_device_properties(0)`（PyTorch 2.9.0+cu128 / CUDA 12.8）。

| 参数 | 实测值 | 在设计中的具体用途 |
|---|---|---|
| GPU 型号 | NVIDIA GeForce RTX 4090 | — |
| Compute Capability | **8.9**（SM89 / Ada Lovelace） | 编译目标 `TORCH_CUDA_ARCH_LIST=8.9`（`w4a16_ops.py`）；决定可用指令集：有 `cp.async`，**无 TMA / 无 wgmma** |
| SM 数量 | **128** | 判断 grid 是否铺得满、以及小 M 时处在一波还是多波。实测 M ≥ 512 时 grid 已铺满全部 128 个 SM，耗时仍与 M=8 持平 → 说明瓶颈不是 SM 数量而是**单 block 的串行 K 循环延迟**（见 §5.2） |
| Warp size | 32 | 线程组织：8 warps = 256 线程，按 4(沿M) × 2(沿N) 划分 |
| 每 SM 最大线程数 | **1536** | `__launch_bounds__(256, 2)` 的合法性上界（2×256 = 512 ≪ 1536，不构成约束） |
| 每 SM 寄存器数 | **65536** | `__launch_bounds__` 第二个参数（每 SM 目标 block 数）的寄存器预算来源；v1/v3 取 2、v2 取 1 |
| **每线程块静态 shared 上限** | **49152 B（48 KB）** | **v1 / v3 的 tile 必须塞进 48 KB**（实测 v1 40 KB、v3 44 KB）；v2 按此上限设计时会直接触发 ptxas 报错 `uses too much shared data (0x14000 bytes, 0xc000 max)` |
| **每线程块动态 shared 上限（opt-in）** | **101376 B（99 KB）** | `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, 99*1024)` 的取值来源（`99*1024 = 101376`，恰好等于本机上限） |
| **每 SM shared memory** | **102400 B（100 KB）** | **决定每 SM 能驻留几个 block，是 v1/v2/v3 性能差异的直接原因**：v1 用 40 KB → 2 blocks/SM；v2 用 80 KB → **只 1 block/SM**（占用率塌陷，0.35x）；v3 用 44 KB → 2 blocks/SM |
| L2 cache | **75497472 B（72 MB）** | 用于判断权重能否在 L2 内复用：int4 权重共 3.6 GB ≫ 72 MB，**任何一层都放不下** → decode 必然流式读 DRAM，故 GEMV 的设计目标定为「最小化 DRAM 流量」而非提高复用 |
| 显存容量 | 25.25 GB | 与 13.598 GB 的 fp16 权重占用对比，量化后省 9.99 GB（§5.3） |

### B.2 规格值（非本机实测，来自 Ada 白皮书 / RTX 4090 官方参数）

这些数字用于**推导性能上界**，不作为 kernel 实现依据：

| 参数 | 规格值 | 在设计中的用途 |
|---|---|---|
| fp16 Tensor Core 峰值算力 | ~165 TFLOPS（fp32 accumulate） | **§6 的核心论据**：本 kernel 与 cuBLAS 用的是同一块硬件单元、同一个上限，因此「追平」的天花板就是 cuBLAS 本身 |
| 显存带宽 | 1008 GB/s（GDDR6X 21 Gbps × 384-bit） | decode roofline：机器平衡点 ≈ 165e12 / 1008e9 ≈ **164 FLOP/B**，据此判定 M=1 时（算术强度 1~4 FLOP/B）是**纯带宽受限**，故 GEMV 只优化访存量 |
| int8 / int4 Tensor Core 峰值 | ~330 TOPS / ~660 TOPS（分别为 fp16 的 2x / 4x） | **§6 中「若能改用 int4 TC 可得 1.4~2x」的收益估算依据** |
| 第 4 代 Tensor Core 指令支持 | `wmma` m16n16k16 fp16、`mma.sync` m16n8k64 **s4.s4.s32**（int4） | 前者是 v1/v2/v3 的实际实现路径；后者是 §6 中「本格式用不上」的那条路 |
| TMA / `wgmma` | **不支持**（Hopper SM90 专属特性） | 排除了 Hopper 风格的大 tile + TMA 方案，只能是 Ada 上 `cp.async` + 双缓冲的路线 |
| `cp.async` | 支持（SM80 起引入） | A 侧异步预取、v2/v3 的 B 侧打包态预取都基于此；其**逐线程** `wait_group` 语义也是 §4.2 那个跨线程可见性 bug 的来源 |

### B.3 由硬件参数直接推出的设计约束

| 约束 | 推导来源 |
|---|---|
| v1 tile 取 128×128×32、每 SM 2 个 block | 每 SM 100 KB shared ÷ 2 block = 50 KB/block，而 48 KB 静态上限下 128×128×32 的配置刚好 40 KB，两者同时满足 |
| v2 必须改用 `extern __shared__` + opt-in | 40KB → 80KB 超出 48 KB 静态上限，只能走 99 KB 的动态共享内存通道 |
| v2 占用率必然掉到 1 block/SM | 80 KB × 2 = 160 KB > 100 KB（每 SM shared 上限），物理上放不下两个 block |
| v3 回退到 BK=32 | 44 KB × 2 = 88 KB ≤ 100 KB，是**在保住 2 blocks/SM 前提下能支持的最大 K 块深度**（BK=64 就超了） |
| 单 block 输出 tile 的 N 宽度锁定 128 | 128 同时满足：① 等于 scale block 宽度，使 block 内 `sblock` 为常量；② 128×128 fp16 的 B tile 能塞进 48 KB |
| decode 用 GEMV 而非 GEMM | 机器平衡 164 FLOP/B 对比 M=1 时的算术强度 1~4 FLOP/B —— 差距 40 倍以上，算力完全无关，只优化带宽 |

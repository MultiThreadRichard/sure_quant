#!/usr/bin/env python
"""Precision + speed microbenchmark for the prefill GEMM: v1 vs v2 vs fp16 cuBLAS."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent          # <repo>/cuda_op -> <repo>
for _p in (str(_HERE), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import w4a16_ops

DEV = "cuda"


def pack_signed_int4(code):
    nib = (code.to(torch.int16) & 0x0F).to(torch.uint8)
    return (nib[..., 0::2] | (nib[..., 1::2] << 4)).contiguous()


def make_weights(K, N, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    code = torch.randint(-8, 7, (K, N), generator=g, dtype=torch.int8).to(DEV)
    scale = (torch.rand(K, N // 128, generator=g) + 0.5).to(DEV)
    return pack_signed_int4(code), scale, code


def ref_gemm(A, code, scale):
    return A.float() @ (code.float() * scale.repeat_interleave(128, dim=1))


def bench(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    shapes = [(8, 4096, 4096), (64, 4096, 4096), (512, 4096, 4096),
              (2048, 4096, 4096), (512, 4096, 11008), (512, 11008, 4096)]

    print(f"{'shape (M,K,N)':<22} {'cuBLAS':>9} {'v1':>9} {'v2':>9} {'v3':>9} {'v1/c':>6} {'v2/c':>6} {'v3/c':>6}")
    print("-" * 82)
    for (M, K, N) in shapes:
        A = torch.randn(M, K, device=DEV, dtype=torch.float16)
        packed, scale, code = make_weights(K, N)
        W_fp16 = (code.float() * scale.repeat_interleave(128, dim=1)).half()

        # correctness (v2/v3)
        ref = ref_gemm(A, code, scale)
        o2 = w4a16_ops.w4a16_gemm_v2(A, packed, scale)
        o3 = w4a16_ops.w4a16_gemm_v3(A, packed, scale)
        rel2 = (o2.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
        rel3 = (o3.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)

        t_cublas = bench(lambda: A @ W_fp16)
        t_v1 = bench(lambda: w4a16_ops.w4a16_gemm(A, packed, scale))
        t_v2 = bench(lambda: w4a16_ops.w4a16_gemm_v2(A, packed, scale))
        t_v3 = bench(lambda: w4a16_ops.w4a16_gemm_v3(A, packed, scale))
        print(f"  {'(%d,%d,%d)' % (M, K, N):<22} {t_cublas:>7.3f} {t_v1:>7.3f} {t_v2:>7.3f} {t_v3:>7.3f} "
              f"{t_cublas / t_v1:>5.2f}x {t_cublas / t_v2:>5.2f}x {t_cublas / t_v3:>5.2f}x   (rel2={rel2:.1e} rel3={rel3:.1e})")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Correctness + performance + memory tests for the SureQuant W4A16 CUDA ops.

Covers the two requirements:
  1) 算子计算正确性  —— GEMM & 逆旋转 epilogue vs torch 参考实现
  2) 模型推理速度 & 显存 —— 自定义 int4 op vs 当前 fp16 (cuBLAS) 路径；int4 vs fp16 权重占用

Run:  python cuda_op/test_w4a16_ops.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent          # <repo>/cuda_op -> <repo>
for _p in (str(_HERE), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import w4a16_ops  # noqa: E402  (JIT-builds the extension on first import)

DEV = "cuda"

# LLaVA-1.5-7B representative linear shapes: (in_features=K, out_features=N)
LANG_SHAPES = [
    ("q/k/v/o_proj", 4096, 4096),
    ("gate/up_proj", 4096, 11008),
    ("down_proj", 11008, 4096),
]
PREFILL_M = [1, 8, 64, 512, 2048]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def pack_signed_int4(code: torch.Tensor) -> torch.Tensor:
    """Mirror of persistence._pack_signed_int4: 2 signed int4 per uint8 (lo=even, hi=odd)."""
    nib = (code.to(torch.int16) & 0x0F).to(torch.uint8)
    return (nib[..., 0::2] | (nib[..., 1::2] << 4)).contiguous()


def make_weights(K: int, N: int, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    code = torch.randint(-8, 7, (K, N), generator=g, dtype=torch.int8).to(DEV)
    scale = (torch.rand(K, N // 128, generator=g) + 0.5).to(DEV)
    packed = pack_signed_int4(code)
    return packed, scale, code


def ref_gemm(A: torch.Tensor, code: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    W_rot = code.float() * scale.repeat_interleave(128, dim=1)   # [K, N]
    return A.float() @ W_rot


def bench(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


# ---------------------------------------------------------------------------
# Part 1: correctness
# ---------------------------------------------------------------------------
def test_correctness():
    print("=" * 72)
    print("Part 1: 算子计算正确性")
    print("=" * 72)

    torch.manual_seed(0)
    shapes = [(128, 32, 128), (16, 4096, 1024), (128, 4096, 4096),
              (1, 4096, 4096), (256, 11008, 4096), (128, 1024, 4096)]
    max_rel = 0.0
    for (M, K, N) in shapes:
        A = torch.randn(M, K, device=DEV, dtype=torch.float16)
        packed, scale, code = make_weights(K, N)
        out = w4a16_ops.w4a16_gemm(A, packed, scale)
        ref = ref_gemm(A, code, scale)
        err = (out.float() - ref).abs().max().item()
        rel = err / (ref.abs().max().item() + 1e-6)
        max_rel = max(max_rel, rel)
        status = "OK" if rel < 2e-2 else "FAIL"
        print(f"  GEMM  M={M:4d} K={K:5d} N={N:5d}: rel={rel:.3e}  [{status}]")
    assert max_rel < 2e-2, f"GEMM correctness failed, max rel={max_rel:.3e}"

    # GEMV (decode) correctness
    for (K, N) in [(4096, 4096), (11008, 4096)]:
        x = torch.randn(K, device=DEV, dtype=torch.float16)
        packed, scale, code = make_weights(K, N)
        y = w4a16_ops.w4a16_gemv(x, packed, scale)
        ref = x.float() @ (code.float() * scale.repeat_interleave(128, dim=1))
        rel = (y.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
        print(f"  GEMV  K={K:5d} N={N:5d}: rel={rel:.3e}  [{'OK' if rel < 2e-2 else 'FAIL'}]")
        assert rel < 2e-2, f"GEMV correctness failed, rel={rel:.3e}"

    # inverse rotation vs CompositeBlockRotation.inverse
    from ops.hadamard import BlockHadamardTransform
    from ops.givens import BlockGivensRotation
    from model.wrappers import CompositeBlockRotation

    torch.manual_seed(1)
    max_rot = 0.0
    for (M, N) in [(4, 1024), (1, 4096), (8, 4096)]:
        nb = N // 128
        had = BlockHadamardTransform(128, nb).to(DEV)
        giv = BlockGivensRotation(128, nb).to(DEV)
        with torch.no_grad():
            giv.theta.uniform_(-0.3, 0.3)
        rot = CompositeBlockRotation(had, giv, order="hadamard_givens").to(DEV)

        z = torch.randn(M, N, device=DEV, dtype=torch.float16)
        y_ref = rot.inverse(z.view(M, nb, 128)).reshape(M, N)
        y_k = w4a16_ops.inverse_rotate(z, had.signs.detach().float(), giv.theta.detach().float())
        err = (y_k.float() - y_ref).abs().max().item()
        rel = err / (y_ref.abs().max().item() + 1e-6)
        max_rot = max(max_rot, rel)
        print(f"  ROT   M={M:4d} N={N:5d}: rel={rel:.3e}  [{'OK' if rel < 5e-3 else 'FAIL'}]")
    assert max_rot < 5e-3, f"inverse_rotate failed, max rel={max_rot:.3e}"

    print("  -> correctness PASSED (GEMM rel<=%.1e, rotate rel<=%.1e)\n" % (max_rel, max_rot))


# ---------------------------------------------------------------------------
# Part 2: speed (custom int4 op vs current fp16 cuBLAS path)
# ---------------------------------------------------------------------------
def test_speed():
    print("=" * 72)
    print("Part 2: 推理速度对比 (custom int4 op vs fp16 cuBLAS)")
    print("=" * 72)

    # ---- decode (M=1): int4 GEMV vs fp16 cuBLAS ----
    print("  [decode, M=1]  int4 GEMV vs fp16 cuBLAS")
    print(f"  {'shape (K,N)':<20} {'fp16 cuBLAS':>12} {'int4 gemv':>12} {'speedup':>9}")
    print("  " + "-" * 54)
    for (K, N) in [(4096, 4096), (4096, 11008), (11008, 4096), (1024, 4096)]:
        x = torch.randn(1, K, device=DEV, dtype=torch.float16)
        packed, scale, code = make_weights(K, N)
        W_fp16 = (code.float() * scale.repeat_interleave(128, dim=1)).half()
        t_fp16 = bench(lambda: x @ W_fp16, 1000)
        t_gemv = bench(lambda: w4a16_ops.w4a16_gemv(x[0], packed, scale), 1000)
        print(f"  {'(%d,%d)' % (K, N):<20} {t_fp16:>10.3f}ms {t_gemv:>10.3f}ms "
              f"{t_fp16 / t_gemv:>8.2f}x")

    # ---- prefill (M>=8): int4 GEMM vs fp16 cuBLAS ----
    print("\n  [prefill, M>1]  int4 GEMM vs fp16 cuBLAS")
    print(f"  {'shape (M,K,N)':<26} {'fp16 cuBLAS':>12} {'int4 gemm':>12} {'speedup':>9}")
    print("  " + "-" * 60)
    for (M, K, N) in [(8, 4096, 4096), (64, 4096, 4096), (512, 4096, 4096),
                      (2048, 4096, 4096), (512, 4096, 11008), (512, 11008, 4096)]:
        A = torch.randn(M, K, device=DEV, dtype=torch.float16)
        packed, scale, code = make_weights(K, N)
        W_fp16 = (code.float() * scale.repeat_interleave(128, dim=1)).half()
        t_fp16 = bench(lambda: A @ W_fp16, 200)
        t_gemm = bench(lambda: w4a16_ops.w4a16_gemm(A, packed, scale), 200)
        print(f"  {'(%d,%d,%d)' % (M, K, N):<26} {t_fp16:>10.3f}ms {t_gemm:>10.3f}ms "
              f"{t_fp16 / t_gemm:>8.2f}x")
    print()


# ---------------------------------------------------------------------------
# Part 3: memory (int4 vs fp16 weight storage)
# ---------------------------------------------------------------------------
def test_memory():
    print("=" * 72)
    print("Part 3: 权重显存占用 (int4 vs fp16)")
    print("=" * 72)

    # per-element bytes
    packed_bpe = 0.5                    # int4: 0.5 B/elem
    scale_bpe = 4.0 / 128.0             # fp32 scale per (k, 128-block): 4/128 B/elem
    fp16_bpe = 2.0                      # fp16: 2 B/elem
    print(f"  每元素: int4 {packed_bpe:.3f} B + scale {scale_bpe:.4f} B = "
          f"{packed_bpe + scale_bpe:.4f} B  vs  fp16 {fp16_bpe} B")
    print(f"  压缩比: {fp16_bpe / (packed_bpe + scale_bpe):.2f}x\n")

    # model-level: sum the real checkpoint's packed int4 weights + scales
    art_path = (REPO_ROOT / "model_saved/llava_7b_surequant_w4a16"
                / "best_quantized_model" / "surequant_int4_weights.pt")
    if art_path.exists():
        art = torch.load(art_path, map_location="cpu", weights_only=True)
        tot_packed = sum(s["packed_weight"].numel() for s in art["layers"].values())
        tot_scale = sum(s["scale"].numel() * 4 for s in art["layers"].values())
        tot_elems = 2 * tot_packed                       # 2 int4 per byte
        fp16_bytes = tot_elems * 2
        int4_bytes = tot_packed + tot_scale              # packed bytes + fp32 scale bytes
        print(f"  量化层数: {len(art['layers'])}")
        print(f"  权重总元素: {tot_elems/1e9:.3f} B")
        print(f"  int4 权重(含 scale): {int4_bytes/1e9:.3f} GB")
        print(f"  fp16 权重:           {fp16_bytes/1e9:.3f} GB")
        print(f"  -> 显存节省: {fp16_bytes - int4_bytes:.0f} B = {(fp16_bytes - int4_bytes)/1e9:.2f} GB "
              f"({fp16_bytes / int4_bytes:.2f}x)\n")


# ---------------------------------------------------------------------------
def main():
    test_correctness()
    test_speed()
    test_memory()
    print("done.")


if __name__ == "__main__":
    main()

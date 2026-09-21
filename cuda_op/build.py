#!/usr/bin/env python
"""Build (cached) + smoke-test the W4A16 custom CUDA ops."""
from __future__ import annotations

import torch

import w4a16_ops


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"
    M, K, N = 16, 4096, 1024

    A = torch.randn(M, K, device=dev, dtype=torch.float16)
    code = torch.randint(-8, 7, (K, N), device=dev, dtype=torch.int8)
    scale = (torch.rand(K, N // 128, device=dev) + 0.5)

    # pack signed int4 (mirror of persistence._pack_signed_int4)
    nib = (code.to(torch.int16) & 0x0F).to(torch.uint8)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()

    out = w4a16_ops.w4a16_gemm(A, packed, scale)
    ref = A.float() @ (code.float() * scale.repeat_interleave(128, dim=1))
    # Metric matches test_w4a16_ops.py: max abs err normalised by the output
    # dynamic range.  A per-element relative metric is unusable here -- with
    # K=4096 random-signed terms the output has heavy cancellation, so |ref| can
    # sit at ~1e-2 while the fp16 roundoff floor is ~4.9e-4 * max|ref|, which
    # makes that ratio blow up for reasons that have nothing to do with the op.
    err = (out.float() - ref).abs().max().item()
    rel = err / (ref.abs().max().item() + 1e-6)
    print(f"[build] w4a16_gemm smoke test: max rel err = {rel:.2e} "
          f"(fp16 unit roundoff = 4.9e-04)")
    assert rel < 2e-2, "smoke test failed"

    # rotation smoke test (block 128, full butterfly)
    Nb = N // 128
    signs = torch.where(torch.rand(Nb, 128, device=dev) > 0.5, 1.0, -1.0)
    theta = torch.randn(Nb, 448, device=dev) * 0.1
    y = w4a16_ops.inverse_rotate(out, signs, theta)
    print(f"[build] inverse_rotate smoke test: out shape {tuple(y.shape)} (no NaN: {not torch.isnan(y).any().item()})")

    print("[build] OK — extension compiled and runs.")


if __name__ == "__main__":
    main()

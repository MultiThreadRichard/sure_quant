"""Loader for the SureQuant W4A16 custom CUDA ops.

Builds (once, cached) and imports the extension produced from ``w4a16_ops.cu``.
Import this module and call :func:`w4a16_gemm` / :func:`inverse_rotate`.
"""
from __future__ import annotations

import os
from pathlib import Path

from torch.utils import cpp_extension

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "w4a16_ops.cu"
_BUILD_DIR = _HERE / "build"

# Only target the installed GPU (RTX 4090 == sm_89) to keep builds fast.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(_BUILD_DIR))

_ext = cpp_extension.load(
    name="w4a16_ops",
    sources=[str(_SRC)],
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)

# --- public API -----------------------------------------------------------
def w4a16_gemm(A, B_packed, scale):
    """int4 x fp16 dequant GEMM.

    A         : [M, K] fp16  (activations)
    B_packed  : [K, N//2] uint8  (signed twos-complement int4, 2/byte)
    scale     : [K, N//128] fp32
    returns   : [M, N] fp16  (pre-rotation output z = A @ W_rot)
    """
    return _ext.w4a16_gemm(A.contiguous(), B_packed.contiguous(), scale.contiguous())


def w4a16_gemm_v2(A, B_packed, scale):
    """v2 int4 x fp16 dequant GEMM (BK=64 pipelined) for compute-bound prefill.

    Same signature as :func:`w4a16_gemm`.
    """
    return _ext.w4a16_gemm_v2(A.contiguous(), B_packed.contiguous(), scale.contiguous())


def w4a16_gemm_v3(A, B_packed, scale):
    """v3 int4 x fp16 dequant GEMM (BK=32 pipelined, 2 blocks/SM).

    Same signature as :func:`w4a16_gemm`.
    """
    return _ext.w4a16_gemm_v3(A.contiguous(), B_packed.contiguous(), scale.contiguous())


def w4a16_gemv(x, B_packed, scale):
    """int4 x fp16 GEMV for decode (M == 1).

    x        : [K] fp16
    B_packed : [K, N//2] uint8
    scale    : [K, N//128] fp32
    returns  : [N] fp16  (pre-rotation z = x @ W_rot)
    """
    y_f32 = _ext.w4a16_gemv(x.contiguous(), B_packed.contiguous(), scale.contiguous())
    return y_f32.half()


def inverse_rotate(z, signs, theta):
    """Apply R^{-1} per 128-wide output block (Givens^-1 then Hadamard^-1).

    z     : [M, N] fp16
    signs : [N//128, 128] fp32  (+-1 hadamard signs buffer)
    theta : [N//128, 448] fp32  (full-butterfly givens angles)
    returns : [M, N] fp16
    """
    theta_cos = theta.cos()
    theta_sin = -theta.sin()
    return _ext.inverse_rotate(
        z.contiguous(), signs.contiguous(), theta_cos.contiguous(), theta_sin.contiguous()
    )

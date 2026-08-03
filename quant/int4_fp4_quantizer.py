"""
INT4 / FP4 量化器复刻实现
================================================
基于 vllm-project/compressed-tensors 仓库底层逻辑复刻
- INT4: 对称/非对称分组量化 + int32 密集打包
- FP4:  NVFP4 E2M1 浮点量化 + uint8 打包 + global_scale
- 演示: 权重量化 → 压缩存储 → 反量化还原 → 误差与压缩分析

依赖: torch >= 2.0
"""

import math
import torch

# ============================================================
# 第一部分: 数据类型常量与定义
# ============================================================

class FloatArgs:
    """浮点格式基类 (对应 compressed_tensors.quant_args.FloatArgs)"""
    exponent: int = 0
    mantissa: int = 0
    bits: int = None
    max: float = None
    min: float = None
    dtype: torch.dtype = None


class FP4_E2M1_DATA(FloatArgs):
    """
    FP4 E2M1 格式: 1位符号 + 2位指数 + 1位尾数 = 4 bit
    有效正值: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    对应 compressed_tensors.quant_args.FP4_E2M1_DATA
    """
    exponent = 2
    mantissa = 1
    bits = 4
    max = 6.0
    min = -6.0
    dtype = None  # FP4 无原生 torch dtype, 用 float32 存储

    # E2M1 查找表: 索引 → 绝对值
    E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

    @staticmethod
    def cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
        """
        将浮点张量舍入到最近的 FP4 E2M1 值.
        对应 compressed_tensors.quant_args.FP4_E2M1_DATA.cast_to_fp4

        舍入区间:
          |x| in [0.00, 0.25] -> 0.0
          |x| in (0.25, 0.75) -> 0.5
          |x| in [0.75, 1.25] -> 1.0
          |x| in (1.25, 1.75) -> 1.5
          |x| in [1.75, 2.50] -> 2.0
          |x| in (2.50, 3.50) -> 3.0
          |x| in [3.50, 5.00] -> 4.0
          |x| > 5.00          -> 6.0
        """
        sign = torch.sign(x)
        x = torch.abs(x)
        x[(x >= 0.0) & (x <= 0.25)] = 0.0
        x[(x > 0.25) & (x < 0.75)] = 0.5
        x[(x >= 0.75) & (x <= 1.25)] = 1.0
        x[(x > 1.25) & (x < 1.75)] = 1.5
        x[(x >= 1.75) & (x <= 2.5)] = 2.0
        x[(x > 2.5) & (x < 3.5)] = 3.0
        x[(x >= 3.5) & (x <= 5.0)] = 4.0
        x[x > 5.0] = 6.0
        return x * sign


class FP8_E4M3_DATA(FloatArgs):
    """
    FP8 E4M3 格式: 1位符号 + 4位指数 + 3位尾数 = 8 bit
    对应 compressed_tensors.quant_args.FP8_E4M3_DATA
    用作 NVFP4 的 per-group scale 存储格式
    """
    exponent = 4
    mantissa = 3
    bits = 8
    max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    min = torch.finfo(torch.float8_e4m3fn).min  # -448.0
    dtype = torch.float8_e4m3fn


# ============================================================
# 第二部分: 量化参数计算
# ============================================================

def calculate_range(num_bits: int, q_type: str, device: torch.device):
    """
    计算量化范围 [q_min, q_max]
    对应 compressed_tensors.quantization.utils.helpers.calculate_range

    :param num_bits: 量化位数
    :param q_type: "int" 或 "float"
    :param device: 目标设备
    :return: (q_min, q_max)
    """
    if q_type == "int":
        bit_range = 2.0 ** num_bits
        q_max = torch.tensor(bit_range / 2 - 1, device=device)  # INT4: 7
        q_min = torch.tensor(-bit_range / 2, device=device)      # INT4: -8
    elif q_type == "float":
        if num_bits == 8:
            q_max = torch.tensor(FP8_E4M3_DATA.max, device=device)
            q_min = torch.tensor(FP8_E4M3_DATA.min, device=device)
        elif num_bits == 4:
            q_max = torch.tensor(FP4_E2M1_DATA.max, device=device)  # 6.0
            q_min = torch.tensor(FP4_E2M1_DATA.min, device=device)  # -6.0
        else:
            raise NotImplementedError("Float quantization only supports 4 or 8 bits")
    else:
        raise ValueError(f"Invalid quantization type: {q_type}")
    return q_min, q_max


def calculate_qparams(
    min_vals: torch.Tensor,
    max_vals: torch.Tensor,
    num_bits: int,
    q_type: str,
    symmetric: bool = True,
    scale_dtype: torch.dtype = None,
    global_scale: torch.Tensor = None,
):
    """
    根据最小/最大值计算 scale 和 zero_point
    对应 compressed_tensors.quantization.utils.helpers.calculate_qparams

    :param min_vals: 观察到的最小值
    :param max_vals: 观察到的最大值
    :param num_bits: 量化位数
    :param q_type: "int" 或 "float"
    :param symmetric: 是否对称量化
    :param scale_dtype: scale 的目标 dtype (FP4 用 FP8)
    :param global_scale: 全局缩放因子 (FP4 专用)
    :return: (scale, zero_point)
    """
    # 确保 0.0 可表示
    min_vals = torch.min(min_vals, torch.zeros_like(min_vals))
    max_vals = torch.max(max_vals, torch.zeros_like(max_vals))

    device = min_vals.device
    q_min, q_max = calculate_range(num_bits, q_type, device)
    bit_range = (q_max - q_min).item()

    if symmetric:
        # 对称量化: scale = max(|min|, |max|) / (bit_range / 2)
        max_val_pos = torch.max(torch.abs(min_vals), torch.abs(max_vals))
        scales = max_val_pos / (float(bit_range) / 2.0)
        zero_points = torch.zeros_like(scales)
    else:
        # 非对称量化 (仅 INT 支持)
        if q_type == "float" and num_bits == 4:
            raise NotImplementedError("Asymmetric quantization is not supported for FP4")
        scales = (max_vals - min_vals) / float(bit_range)
        zero_points = q_min - (min_vals / scales)
        zero_points = torch.clamp(zero_points, q_min.item(), q_max.item())

    # 应用全局缩放 (FP4)
    if global_scale is not None:
        scales = global_scale * scales

    # 舍入 scale 到指定 dtype
    if scale_dtype is not None:
        if torch.is_floating_point(torch.tensor([], dtype=scale_dtype)):
            finfo = torch.finfo(scale_dtype)
            scales = torch.clamp(scales, finfo.min, finfo.max).to(scale_dtype).to(torch.float32)
        else:
            iinfo = torch.iinfo(scale_dtype)
            scales = torch.round(torch.clamp(scales, iinfo.min, iinfo.max)).to(scale_dtype).to(torch.float32)

    # 防零除
    eps = 1e-8 if scales.dtype == torch.float32 else 0.125
    scales = torch.where(scales == 0, torch.tensor(eps, dtype=scales.dtype, device=device), scales)

    if scales.ndim == 0:
        scales = scales.reshape(1)
        zero_points = zero_points.reshape(1)

    return scales, zero_points


def generate_global_scale(
    min_vals: torch.Tensor,
    max_vals: torch.Tensor,
    scale_max: float = FP8_E4M3_DATA.max,   # 448.0
    quant_max: float = FP4_E2M1_DATA.max,   # 6.0
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    为整个张量生成全局缩放因子 (NVFP4 专用)
    对应 compressed_tensors.quantization.utils.helpers.generate_gparam

    公式: global_scale = scale_max * quant_max / max(|min|, |max|)
    作用: 确保 per-group FP8 scale 能充分利用 FP8 动态范围

    :return: shape=[1] 的全局缩放张量
    """
    min_vals = torch.min(min_vals, torch.zeros_like(min_vals))
    max_vals = torch.max(max_vals, torch.zeros_like(max_vals))
    max_val_pos = torch.max(torch.abs(min_vals), torch.abs(max_vals))
    max_val_pos = torch.clamp(max_val_pos, min=torch.finfo(max_val_pos.dtype).tiny)

    global_scale = scale_max * quant_max / max_val_pos
    global_scale = torch.nan_to_num(global_scale, nan=1.0, posinf=1.0, neginf=1.0)
    return global_scale.to(dtype).reshape([1])


# ============================================================
# 第三部分: 量化 / 反量化核心函数
# ============================================================

def _round_to_quantized_type(tensor, q_type, num_bits, q_min, q_max):
    """
    将浮点张量舍入到量化类型
    对应 compressed_tensors.quant_args.round_to_quantized_type_args
    """
    tensor = torch.clamp(tensor, q_min, q_max)
    if q_type == "float":
        if num_bits == 8:
            return tensor.to(FP8_E4M3_DATA.dtype).to(torch.float32)
        elif num_bits == 4:
            return FP4_E2M1_DATA.cast_to_fp4(tensor)
    elif q_type == "int":
        return torch.round(tensor)
    return tensor


@torch.no_grad()
def quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    num_bits: int,
    q_type: str,
    group_size: int = None,
    dtype: torch.dtype = None,
    global_scale: torch.Tensor = None,
) -> torch.Tensor:
    """
    量化函数 — 支持 tensor/channel/group 策略
    对应 compressed_tensors.quantization.lifecycle.forward.quantize + _process_group + _quantize

    :param x: 输入权重张量 (FP16/BF16/FP32)
    :param scale: 缩放因子 (per-tensor: [1], per-group: [rows, num_groups])
    :param zero_point: 零点 (对称量化时为 None)
    :param num_bits: 量化位数
    :param q_type: "int" 或 "float"
    :param symmetric: 是否对称量化
    :param group_size: 分组大小 (None 表示 per-tensor/channel)
    :param dtype: 输出 dtype (INT4 → int8)
    :param global_scale: 全局缩放 (FP4 专用)
    :return: 量化后的张量
    """
    q_min, q_max = calculate_range(num_bits, q_type, x.device)

    # 计算 effective scale
    effective_scale = scale
    if global_scale is not None:
        effective_scale = scale / global_scale

    if group_size is not None and group_size > 0:
        # ---- 分组量化 ----
        columns = x.shape[-1]
        num_groups = math.ceil(columns / group_size)

        # 确保 scale 是 2D
        while effective_scale.ndim < 2:
            effective_scale = effective_scale.unsqueeze(1)
        if zero_point is not None:
            while zero_point.ndim < 2:
                zero_point = zero_point.unsqueeze(1)

        # reshape: (..., columns) -> (..., num_groups, group_size)
        x_grouped = x.unflatten(-1, (num_groups, group_size))
        scale_expanded = effective_scale.unsqueeze(-1)  # (..., num_groups, 1)
        zp_expanded = zero_point.unsqueeze(-1) if zero_point is not None else None

        # 量化: x / scale + zp → clamp → round
        scaled = x_grouped / scale_expanded
        if zp_expanded is not None:
            scaled = scaled + zp_expanded.to(x.dtype)

        quantized = _round_to_quantized_type(scaled, q_type, num_bits, q_min, q_max)

        # flatten 回原始维度
        quantized = quantized.flatten(start_dim=-2)
    else:
        scaled = x / effective_scale
        if zero_point is not None:
            scaled = scaled + zero_point.to(x.dtype)
        quantized = _round_to_quantized_type(scaled, q_type, num_bits, q_min, q_max)

    if dtype is not None:
        quantized = quantized.to(dtype)

    return quantized


@torch.no_grad()
def dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor = None,
    group_size: int = None,
    dtype: torch.dtype = None,
    global_scale: torch.Tensor = None,
) -> torch.Tensor:
    """
    反量化函数 — 支持 tensor/channel/group 策略
    对应 compressed_tensors.quantization.lifecycle.forward.dequantize + _dequantize

    :param x_q: 量化后的张量 (int8 或 float)
    :param scale: 缩放因子
    :param zero_point: 零点
    :param group_size: 分组大小
    :param dtype: 输出 dtype
    :param global_scale: 全局缩放 (FP4 专用)
    :return: 反量化后的浮点张量
    """
    # 计算 effective scale
    effective_scale = scale
    if global_scale is not None:
        effective_scale = scale / global_scale

    if group_size is not None and group_size > 0:
        # ---- 分组反量化 ----
        columns = x_q.shape[-1]
        num_groups = math.ceil(columns / group_size)

        while effective_scale.ndim < 2:
            effective_scale = effective_scale.unsqueeze(1)
        if zero_point is not None:
            while zero_point.ndim < 2:
                zero_point = zero_point.unsqueeze(1)

        x_grouped = x_q.unflatten(-1, (num_groups, group_size))
        scale_expanded = effective_scale.unsqueeze(-1)
        zp_expanded = zero_point.unsqueeze(-1) if zero_point is not None else None

        dequant = x_grouped.to(scale_expanded.dtype)
        if zp_expanded is not None:
            dequant = dequant - zp_expanded.to(scale_expanded.dtype)
        dequant = dequant * scale_expanded
        dequant = dequant.flatten(start_dim=-2)
    else:
        # ---- per-tensor / per-channel 反量化 ----
        dequant = x_q.to(effective_scale.dtype)
        if zero_point is not None:
            dequant = dequant - zero_point.to(effective_scale.dtype)
        dequant = dequant * effective_scale

    if dtype is not None:
        dequant = dequant.to(dtype)

    return dequant


# ============================================================
# 第四部分: INT4 打包/解包 (int32 密集打包)
# ============================================================

def pack_to_int32(value: torch.Tensor, num_bits: int = 4, packed_dim: int = 1) -> torch.Tensor:
    """
    将 int8 量化值密集打包为 int32
    对应 compressed_tensors.compressors.pack_quantized.helpers.pack_to_int32

    INT4: 8 个 4-bit 值 → 1 个 int32, 无填充位
    跨 int32 边界的值会被正确拆分

    :param value: int8 量化张量 (值域 [-8, 7])
    :param num_bits: 每元素位数 (4 for INT4)
    :param packed_dim: 打包维度 (0 或 1)
    :return: int32 打包张量
    """
    assert value.dtype == torch.int8, "Tensor must be torch.int8"
    assert 1 <= num_bits <= 8

    # N 维张量递归处理
    if value.ndim > 2:
        return torch.stack([pack_to_int32(value[i], num_bits, packed_dim)
                            for i in range(value.shape[0])])

    # 转为无符号范围: [-8,7] -> [0,15]
    offset = 1 << (num_bits - 1)  # 8 for 4-bit
    value = value.to(torch.int32) + offset
    device = value.device

    if packed_dim == 0:
        value = value.transpose(0, 1)

    rows, cols = value.shape
    packed_cols = math.ceil(cols * num_bits / 32)  # INT4: ceil(cols*4/32) = ceil(cols/8)

    # 填充到 32 的倍数以便 reshape
    padded_cols = math.ceil(cols / 32) * 32
    if padded_cols > cols:
        value = torch.nn.functional.pad(value, (0, padded_cols - cols))

    num_groups = padded_cols // 32
    rows_g = rows * num_groups
    value_g = value.reshape(rows_g, 32)

    # 每组 32 个元素打包进 num_bits 个 int32
    output_g = torch.zeros(rows_g, num_bits, dtype=torch.int32, device=device)

    elem_i = torch.arange(32, device=device, dtype=torch.int32)
    bit_starts = elem_i * num_bits
    word_idx = (bit_starts // 32).long()
    bit_offset = bit_starts % 32

    # 主打包: 每个元素左移到对应位偏移
    output_g.scatter_add_(
        1,
        word_idx.unsqueeze(0).expand(rows_g, -1),
        value_g << bit_offset.unsqueeze(0),
    )

    # 处理跨 int32 边界的溢出位
    ov = bit_offset + num_bits - 32
    ov_mask = ov > 0
    if ov_mask.any():
        ov_vals = value_g[:, ov_mask] >> (num_bits - ov[ov_mask]).unsqueeze(0)
        output_g.scatter_add_(
            1,
            (word_idx[ov_mask] + 1).unsqueeze(0).expand(rows_g, -1),
            ov_vals,
        )

    # 截断到精确需要的 int32 数量
    output = output_g.view(rows, num_groups * num_bits)[:, :packed_cols]

    if packed_dim == 0:
        output = output.transpose(0, 1)

    return output


def unpack_from_int32(
    value: torch.Tensor,
    num_bits: int = 4,
    shape: torch.Size = None,
    packed_dim: int = 1,
) -> torch.Tensor:
    """
    从 int32 解包还原 int8 值
    对应 compressed_tensors.compressors.pack_quantized.helpers.unpack_from_int32

    :param value: int32 打包张量
    :param num_bits: 每元素位数
    :param shape: 原始形状 (用于确定元素数量)
    :param packed_dim: 打包维度
    :return: int8 解包张量 (值域 [-8, 7])
    """
    assert value.dtype == torch.int32
    assert 1 <= num_bits <= 8

    if value.ndim > 2:
        return torch.stack([unpack_from_int32(value[i], num_bits, shape[1:], packed_dim)
                            for i in range(value.shape[0])])

    if packed_dim == 0:
        value = value.transpose(0, 1)

    rows, num_words = value.shape
    if shape is not None:
        cols = int(shape[packed_dim]) if not isinstance(shape, torch.Tensor) else int(shape[packed_dim].item())
    else:
        cols = num_words * 32 // num_bits

    # 填充到 num_bits 的倍数
    if num_words % num_bits != 0:
        pad_words = num_bits - (num_words % num_bits)
        value = torch.nn.functional.pad(value, (0, pad_words))
        num_words += pad_words

    num_groups = num_words // num_bits
    rows_g = rows * num_groups
    value_g = value.reshape(rows_g, num_bits)

    elem_i = torch.arange(32, device=value.device, dtype=torch.int32)
    bit_starts = elem_i * num_bits
    word_idx = (bit_starts // 32).long()
    bit_offset = bit_starts % 32
    lo_bits = torch.clamp(32 - bit_offset, max=num_bits)

    # 提取低位
    output_g = (value_g[:, word_idx] >> bit_offset.unsqueeze(0)) & \
               ((1 << lo_bits) - 1).unsqueeze(0)

    # 处理跨 int32 边界的值
    ov_mask = lo_bits < num_bits
    if ov_mask.any():
        hi_bits = num_bits - lo_bits[ov_mask]
        right = (value_g[:, word_idx[ov_mask] + 1] & ((1 << hi_bits) - 1).unsqueeze(0)) \
                << lo_bits[ov_mask].unsqueeze(0)
        output_g[:, ov_mask] |= right

    output = output_g.view(rows, num_groups * 32)[:, :cols]

    if packed_dim == 0:
        output = output.transpose(0, 1)

    # 还原有符号范围: [0,15] -> [-8,7]
    offset = 1 << (num_bits - 1)
    return (output - offset).to(torch.int8)


# ============================================================
# 第五部分: FP4 打包/解包 (uint8 打包)
# ============================================================

# E2M1 查找表常量
_E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def pack_fp4_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """
    将 FP4 浮点值打包为 uint8 (2 个 FP4 → 1 个 uint8)
    对应 compressed_tensors.compressors.nvfp4.helpers.pack_fp4_to_uint8

    :param x: 浮点张量, 值在 FP4 有效范围内 [-6.0, 6.0]
    :return: uint8 打包张量, 形状 (...shape[:-1], shape[-1]//2)
    """
    # 支持任意维度的输入，只处理最后一维
    original_shape = x.shape
    assert original_shape[-1] % 2 == 0, "last dimension must be even for fp4 packing"

    # 展平为 2D 进行处理
    m = x.numel() // original_shape[-1]
    n = original_shape[-1]
    x_2d = x.reshape(m, n)

    device = x.device
    lut = _E2M1_LUT.to(device=device, dtype=x.dtype)

    # 找到每个元素最近的 FP4 值的索引 (0-7)
    abs_x = torch.abs(x_2d)
    abs_diff = torch.abs(abs_x.unsqueeze(-1) - lut)  # [m, n, 8]
    abs_indices = torch.argmin(abs_diff, dim=-1)       # [m, n]

    # 应用符号位 (bit 3): 4-bit 表示 [0, 15]
    indices = abs_indices + (torch.signbit(x_2d).to(torch.long) << 3)

    # 相邻两个 4-bit 值打包进一个 uint8
    indices = indices.reshape(-1, 2)
    packed = (indices[:, 0] | (indices[:, 1] << 4)).to(torch.uint8)

    # 恢复原始形状（最后一维减半）
    new_shape = original_shape[:-1] + (original_shape[-1] // 2,)
    return packed.reshape(new_shape)


def unpack_fp4_from_uint8(
    a: torch.Tensor,
    original_shape: tuple,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    从 uint8 解包还原 FP4 浮点值
    对应 compressed_tensors.compressors.nvfp4.helpers.unpack_fp4_from_uint8

    :param a: uint8 打包张量
    :param original_shape: 原始张量形状 (解包后的形状)
    :param dtype: 输出 dtype
    :return: FP4 浮点张量, 形状与 original_shape 一致
    """
    assert a.dtype == torch.uint8

    # 展平为 1D 进行处理
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4  # 高 nibble
    low = a_flat & 0x0F            # 低 nibble

    # 交替组合: [low_0, high_0, low_1, high_1, ...]
    combined = torch.stack((low, high), dim=1).flatten()

    # 提取符号位 (bit 3) 和幅度索引 (bit 0-2)
    signs = (combined & 0x08).to(torch.bool)
    abs_vals = (combined & 0x07).to(torch.long)

    # 查表得到绝对值, 应用符号
    lut = _E2M1_LUT.to(device=a.device)
    values = lut[abs_vals] * torch.where(signs, -1.0, 1.0)

    return values.reshape(original_shape).to(dtype=dtype)


# ============================================================
# 第六部分: 完整量化器封装
# ============================================================

class INT4Quantizer:
    """
    INT4 量化器 — 复刻 PackedQuantizationCompressor 的完整流程

    支持:
    - 对称/非对称量化
    - per-tensor / per-channel / per-group 策略
    - int32 密集打包 (8 个 4-bit 值 / 1 个 int32)
    """

    def __init__(self, group_size: int = 128, symmetric: bool = True):
        self.num_bits = 4
        self.q_type = "int"
        self.group_size = group_size
        self.symmetric = symmetric

    def calibrate(self, weight: torch.Tensor):
        """计算量化参数 (scale, zero_point)"""
        if self.group_size and self.group_size > 0:
            # per-group: 按 group_size 分组计算 min/max
            cols = weight.shape[-1]
            num_groups = math.ceil(cols / self.group_size)
            # 填充到 group_size 的倍数
            pad = num_groups * self.group_size - cols
            if pad > 0:
                weight = torch.nn.functional.pad(weight, (0, pad))
            weight_grouped = weight.unflatten(-1, (num_groups, self.group_size))
            min_val = weight_grouped.amin(dim=-1)  # (..., num_groups)
            max_val = weight_grouped.amax(dim=-1)
        else:
            # per-tensor
            min_val = weight.amin()
            max_val = weight.amax()

        scale, zp = calculate_qparams(
            min_val, max_val,
            num_bits=self.num_bits,
            q_type=self.q_type,
            symmetric=self.symmetric,
        )
        return scale, zp

    def quantize(self, weight: torch.Tensor):
        """量化 → 打包, 返回压缩后的字典"""
        scale, zp = self.calibrate(weight)

        # 量化到 int8 (值域 [-8, 7])
        quantized = quantize(
            weight, scale, zp if not self.symmetric else None,
            num_bits=self.num_bits, q_type=self.q_type,
            group_size=self.group_size,
            dtype=torch.int8,
        )

        # 打包为 int32
        packed = pack_to_int32(quantized, num_bits=self.num_bits)

        return {
            "weight_packed": packed,
            "weight_scale": scale,
            "weight_zero_point": zp if not self.symmetric else None,
            "weight_shape": torch.tensor(weight.shape),
            "original_dtype": weight.dtype,
        }

    def dequantize(self, compressed: dict) -> torch.Tensor:
        """解包 → 反量化"""
        packed = compressed["weight_packed"]
        scale = compressed["weight_scale"]
        zp = compressed.get("weight_zero_point", None)
        original_shape = compressed["weight_shape"]

        # 解包
        unpacked = unpack_from_int32(
            packed, num_bits=self.num_bits, shape=original_shape
        )

        # 反量化
        dequant = dequantize(
            unpacked, scale, zp if not self.symmetric else None,
            group_size=self.group_size,
            dtype=compressed.get("original_dtype", torch.float32),
        )

        # 截断到原始形状 (去除填充)
        return dequant[:original_shape[0], :original_shape[1]]


# class FP4Quantizer:
#     """
#     FP4 (NVFP4 E2M1) 量化器 — 复刻 NVFP4PackedCompressor 的完整流程

#     支持:
#     - FP4 E2M1 浮点量化 (对称)
#     - per-group scale (默认 group_size=16)
#     - global_scale (FP8 范围映射)
#     - uint8 打包 (2 个 FP4 值 / 1 个 uint8)
#     """

#     def __init__(self, group_size: int = 16, pack: bool = False):
#         self.num_bits = 4
#         self.q_type = "float"
#         self.group_size = group_size
#         self.symmetric = True
#         self.pack = pack  # 控制是否进行FP4打包/解包

#     def calibrate(self, weight: torch.Tensor):
#         """计算 global_scale 和 per-group scale"""
#         # 1. 生成 global_scale
#         if self.group_size and self.group_size > 0:
#             cols = weight.shape[-1]
#             num_groups = math.ceil(cols / self.group_size)
#             pad = num_groups * self.group_size - cols
#             if pad > 0:
#                 weight = torch.nn.functional.pad(weight, (0, pad))
#             weight_grouped = weight.unflatten(-1, (num_groups, self.group_size))
#             min_val = weight_grouped.amin(dim=-1)
#             max_val = weight_grouped.amax(dim=-1)
#         else:
#             min_val = weight.amin()
#             max_val = weight.amax()

#         # global_scale: 标量, shape=[1]
#         global_min = min_val.amin()
#         global_max = max_val.amax()
#         global_scale = generate_global_scale(global_min, global_max)

#         # 2. 计算 per-group scale (会被 global_scale 缩放)
#         scale, _ = calculate_qparams(
#             min_val, max_val,
#             num_bits=self.num_bits,
#             q_type=self.q_type,
#             symmetric=self.symmetric,
#             scale_dtype=torch.float8_e4m3fn,  # scale 存储为 FP8
#             global_scale=global_scale,
#         )

#         return global_scale, scale

#     def quantize(self, weight: torch.Tensor):
#         """量化 → 打包(可选), 返回压缩后的字典"""
#         # 确保列数为偶数 (FP4 打包要求)
#         cols = weight.shape[-1]
#         pad = cols % 2
#         if pad > 0:
#             weight = torch.nn.functional.pad(weight, (0, pad))
        
#         original_shape = weight.shape

#         global_scale, scale = self.calibrate(weight)

#         # 量化到 FP4 E2M1
#         quantized = quantize(
#             weight, scale, None,
#             num_bits=self.num_bits, q_type=self.q_type,
#             symmetric=self.symmetric, group_size=self.group_size,
#             global_scale=global_scale,
#         )

#         # 根据pack属性决定是否打包为 uint8
#         if self.pack:
#             packed = pack_fp4_to_uint8(quantized)
#         else:
#             packed = quantized  # 不打包，直接存储量化后的FP4值

#         return {
#             "weight_packed": packed,
#             "weight_scale": scale,                    # FP8 精度的 scale (float32表示)
#             "weight_global_scale": global_scale,      # 全局缩放
#             "weight_shape": original_shape,           # 权重形状 (包含填充)
#             "original_cols": cols,                  # 原始列数 (不填充)
#             "original_dtype": weight.dtype,
#         }

#     def dequantize(self, compressed: dict) -> torch.Tensor:
#         """解包(可选) → 反量化"""
#         packed = compressed["weight_packed"]
#         scale = compressed["weight_scale"]
#         global_scale = compressed["weight_global_scale"]
#         original_shape = compressed["weight_shape"]
#         original_cols = compressed["original_cols"]

#         # 根据pack属性决定是否解包
#         if self.pack:
#             # 解包为 FP4 浮点值
#             unpacked = unpack_fp4_from_uint8(packed, original_shape, dtype=torch.float32)
#         else:
#             # 未打包，直接使用
#             unpacked = packed

#         # 反量化
#         dequant = dequantize(
#             unpacked, scale, None,
#             group_size=self.group_size,
#             dtype=compressed.get("original_dtype", torch.float32),
#             global_scale=global_scale,
#         )

#         # 截断到原始列数 (去除填充)
#         return dequant[..., :original_cols]


class FP4SingleQuantizer:
    """
    FP4 (NVFP4 E2M1) 量化器 — 不使用 global_scale 的简化版本

    与 FP4Quantizer 的主要区别:
    - 不使用 global_scale 进行范围映射
    - scale 直接存储为 float32 而非 FP8
    - 量化精度可能略有损失，但简化了计算流程

    支持:
    - FP4 E2M1 浮点量化 (对称)
    - per-group scale (默认 group_size=16)
    - uint8 打包 (2 个 FP4 值 / 1 个 uint8)
    """

    def __init__(self, group_size: int = 16, pack: bool = False):
        self.num_bits = 4
        self.q_type = "float"
        self.group_size = group_size
        self.symmetric = True
        self.pack = pack  # 控制是否进行FP4打包/解包

    def calibrate(self, weight: torch.Tensor):
        """计算 per-group scale (不使用 global_scale)"""
        if self.group_size and self.group_size > 0:
            cols = weight.shape[-1]
            num_groups = math.ceil(cols / self.group_size)
            pad = num_groups * self.group_size - cols
            if pad > 0:
                weight = torch.nn.functional.pad(weight, (0, pad))
            weight_grouped = weight.unflatten(-1, (num_groups, self.group_size))
            min_val = weight_grouped.amin(dim=-1)
            max_val = weight_grouped.amax(dim=-1)
        else:
            min_val = weight.amin()
            max_val = weight.amax()

        # 直接计算 per-group scale (不使用 global_scale)
        # 对称量化: scale = max(|min|, |max|) / 6.0
        scale, _ = calculate_qparams(
            min_val, max_val,
            num_bits=self.num_bits,
            q_type=self.q_type,
            symmetric=self.symmetric,
            scale_dtype=None,  # 不使用 FP8 存储
            global_scale=None,  # 不使用 global_scale
        )

        return scale

    def quantize(self, weight: torch.Tensor):
        """量化 → 打包(可选), 返回压缩后的字典"""
        # 确保列数为偶数 (FP4 打包要求)
        cols = weight.shape[-1]
        pad = cols % 2
        if pad > 0:
            weight = torch.nn.functional.pad(weight, (0, pad))
        
        original_shape = weight.shape

        scale = self.calibrate(weight)

        # 量化到 FP4 E2M1 (不使用 global_scale)
        quantized = quantize(
            weight, scale, None,
            num_bits=self.num_bits, q_type=self.q_type,
            group_size=self.group_size,
            global_scale=None,  # 不使用 global_scale
        )

        # 根据pack属性决定是否打包为 uint8
        if self.pack:
            packed = pack_fp4_to_uint8(quantized)
        else:
            packed = quantized  # 不打包，直接存储量化后的FP4值

        return {
            "weight_packed": packed,
            "weight_scale": scale,                    # float32 精度的 scale
            "weight_shape": original_shape,           # 权重形状 (包含填充)
            "original_cols": cols,                  # 原始列数 (不填充)
            "original_dtype": weight.dtype,
        }

    def dequantize(self, compressed: dict) -> torch.Tensor:
        """解包(可选) → 反量化"""
        packed = compressed["weight_packed"]
        scale = compressed["weight_scale"]
        original_shape = compressed["weight_shape"]
        original_cols = compressed["original_cols"]

        # 根据pack属性决定是否解包
        if self.pack:
            # 解包为 FP4 浮点值
            unpacked = unpack_fp4_from_uint8(packed, original_shape, dtype=torch.float32)
        else:
            # 未打包，直接使用
            unpacked = packed

        # 反量化 (不使用 global_scale)
        dequant = dequantize(
            unpacked, scale, None,
            group_size=self.group_size,
            dtype=compressed.get("original_dtype", torch.float32),
            global_scale=None,  # 不使用 global_scale
        )

        # 截断到原始列数 (去除填充)
        return dequant[..., :original_cols]


# # ============================================================
# # 第七部分: 演示主程序
# # ============================================================

# def demo_int4():
#     """INT4 量化演示"""
#     print("=" * 70)
#     print("INT4 量化演示 (对称 + per-group, group_size=128)")
#     print("=" * 70)

#     # 生成模拟权重 (模拟 LLaMA 线性层权重分布)
#     torch.manual_seed(42)
#     weight = torch.randn(256, 512, dtype=torch.float32) * 0.1
#     # 添加一些 outlier 模拟真实分布
#     weight[0, ::64] += 2.0
#     weight[::32, 0] -= 1.5

#     print(f"\n原始权重: shape={weight.shape}, dtype={weight.dtype}")
#     print(f"  范围: [{weight.min():.4f}, {weight.max():.4f}]")
#     print(f"  原始大小: {weight.nelement() * 4} bytes ({weight.nelement() * 4 / 1024:.1f} KB)")

#     # 量化
#     quantizer = INT4Quantizer(group_size=128, symmetric=True)
#     compressed = quantizer.quantize(weight)

#     packed = compressed["weight_packed"]
#     scale = compressed["weight_scale"]

#     print(f"\n压缩后:")
#     print(f"  weight_packed: shape={packed.shape}, dtype={packed.dtype}")
#     print(f"  weight_scale:  shape={scale.shape}, dtype={scale.dtype}")
#     print(f"  packed 大小: {packed.nelement() * 4} bytes")
#     print(f"  scale 大小:   {scale.nelement() * 4} bytes")
#     total_compressed = packed.nelement() * 4 + scale.nelement() * 4
#     total_original = weight.nelement() * 4
#     print(f"  总压缩大小:   {total_compressed} bytes")
#     print(f"  压缩比:      {total_original / total_compressed:.2f}x")

#     # 反量化
#     dequantized = quantizer.dequantize(compressed)

#     # 误差分析
#     error = (weight - dequantized)
#     mse = torch.mean(error ** 2).item()
#     max_error = torch.max(torch.abs(error)).item()
#     cos_sim = torch.nn.functional.cosine_similarity(
#         weight.flatten().unsqueeze(0),
#         dequantized.flatten().unsqueeze(0)
#     ).item()

#     print(f"\n反量化结果:")
#     print(f"  shape={dequantized.shape}, dtype={dequantized.dtype}")
#     print(f"  MSE:          {mse:.8f}")
#     print(f"  Max |error|:  {max_error:.6f}")
#     print(f"  Cosine Sim:   {cos_sim:.8f}")
#     print(f"  相对 L2 误差: {torch.norm(error) / torch.norm(weight):.6f}")

#     # 验证打包/解包的正确性 (不含量化误差)
#     print(f"\n打包/解包验证 (无量化误差):")
#     test_int8 = torch.randint(-8, 7, (4, 512), dtype=torch.int8)
#     test_packed = pack_to_int32(test_int8, num_bits=4)
#     test_unpacked = unpack_from_int32(test_packed, num_bits=4, shape=test_int8.shape)
#     pack_match = torch.equal(test_int8, test_unpacked)
#     print(f"  输入: {test_int8.shape} (int8) → 打包: {test_packed.shape} (int32) → 解包: {test_unpacked.shape} (int8)")
#     print(f"  精确还原: {'✅ PASS' if pack_match else '❌ FAIL'}")

#     print()


# def demo_fp4():
#     """FP4 (NVFP4 E2M1) 量化演示"""
#     print("=" * 70)
#     print("FP4 (NVFP4 E2M1) 量化演示 (对称 + per-group, group_size=16)")
#     print("=" * 70)

#     # 生成模拟权重
#     torch.manual_seed(42)
#     # weight = torch.randn(256, 512, dtype=torch.float32) * 0.1
#     weight = torch.randn(256, 511, dtype=torch.float32) * 0.1

#     weight[0, ::64] += 2.0
#     weight[::32, 0] -= 1.5

#     print(f"\n原始权重: shape={weight.shape}, dtype={weight.dtype}")
#     print(f"  范围: [{weight.min():.4f}, {weight.max():.4f}]")
#     print(f"  原始大小: {weight.nelement() * 4} bytes ({weight.nelement() * 4 / 1024:.1f} KB)")

#     # 量化
#     quantizer = FP4Quantizer(group_size=16)
#     compressed = quantizer.quantize(weight)

#     packed = compressed["weight_packed"]
#     scale = compressed["weight_scale"]
#     global_scale = compressed["weight_global_scale"]

#     print(f"\n压缩后:")
#     print(f"  weight_packed:      shape={packed.shape}, dtype={packed.dtype}")
#     print(f"  weight_scale:       shape={scale.shape}, dtype={scale.dtype}")
#     print(f"  weight_global_scale: shape={global_scale.shape}, value={global_scale.item():.4f}")
#     print(f"  packed 大小: {packed.nelement() * 1} bytes (uint8)")
#     print(f"  scale 大小:  {scale.nelement() * 4} bytes")
#     print(f"  global 大小: {global_scale.nelement() * 4} bytes")
#     total_compressed = packed.nelement() * 1 + scale.nelement() * 4 + global_scale.nelement() * 4
#     total_original = weight.nelement() * 4
#     print(f"  总压缩大小:  {total_compressed} bytes")
#     print(f"  压缩比:     {total_original / total_compressed:.2f}x")

#     # 反量化
#     dequantized = quantizer.dequantize(compressed)

#     # 误差分析
#     error = (weight - dequantized)
#     mse = torch.mean(error ** 2).item()
#     max_error = torch.max(torch.abs(error)).item()
#     cos_sim = torch.nn.functional.cosine_similarity(
#         weight.flatten().unsqueeze(0),
#         dequantized.flatten().unsqueeze(0)
#     ).item()

#     print(f"\n反量化结果:")
#     print(f"  shape={dequantized.shape}, dtype={dequantized.dtype}")
#     print(f"  MSE:          {mse:.8f}")
#     print(f"  Max |error|:  {max_error:.6f}")
#     print(f"  Cosine Sim:   {cos_sim:.8f}")
#     print(f"  相对 L2 误差: {torch.norm(error) / torch.norm(weight):.6f}")

#     # 验证 FP4 打包/解包
#     print(f"\nFP4 打包/解包验证:")
#     test_fp4 = torch.tensor([
#         [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
#         [-0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0],
#     ], dtype=torch.float32)
#     test_packed = pack_fp4_to_uint8(test_fp4)
#     test_unpacked = unpack_fp4_from_uint8(test_packed, test_fp4.shape, dtype=torch.float32)
#     print(f"  输入:   {test_fp4[0].tolist()}")
#     print(f"  打包:   {test_packed[0].tolist()}")
#     print(f"  解包:   {test_unpacked[0].tolist()}")
#     fp4_match = torch.allclose(test_fp4, test_unpacked, atol=1e-6)
#     print(f"  精确还原: {'✅ PASS' if fp4_match else '❌ FAIL'}")

#     print()


# def demo_fp4_no_global_scale():
#     """FP4NoGlobalScaleQuantizer 测试演示 — 对比有无 global_scale 的效果"""
#     print("=" * 70)
#     print("FP4NoGlobalScaleQuantizer 测试演示")
#     print("=" * 70)

#     # 生成模拟权重 (模拟真实 LLM 权重分布)
#     torch.manual_seed(42)
#     weight = torch.randn(256, 512, dtype=torch.float32) * 0.1
#     # weight = torch.randn(256, 511, dtype=torch.float32) * 0.1

#     weight[0, ::64] += 2.0
#     weight[::32, 0] -= 1.5

#     print(f"\n原始权重: shape={weight.shape}, dtype={weight.dtype}")
#     print(f"  范围: [{weight.min():.4f}, {weight.max():.4f}]")
#     print(f"  原始大小: {weight.nelement() * 4} bytes ({weight.nelement() * 4 / 1024:.1f} KB)")

#     # 对比: 有 global_scale vs 无 global_scale
#     results = []
    
#     for name, quantizer in [
#         ("FP4 (无 global_scale)", FP4SingleQuantizer(group_size=16)),
#         ("FP4 (无 global_scale, pack)", FP4SingleQuantizer(group_size=16, pack=True)),
#     ]:
#         compressed = quantizer.quantize(weight)
#         dequantized = quantizer.dequantize(compressed)

#         print(f"\n反量化结果:")
#         print(f"  shape={dequantized.shape}, dtype={dequantized.dtype}")

#         error = weight - dequantized
#         mse = torch.mean(error ** 2).item()
#         cos_sim = torch.nn.functional.cosine_similarity(
#             weight.flatten().unsqueeze(0),
#             dequantized.flatten().unsqueeze(0)
#         ).item()
#         rel_l2 = (torch.norm(error) / torch.norm(weight)).item()
#         # 统计零值比例
#         zero_ratio = (dequantized == 0).float().mean().item() * 100

#         # 计算压缩大小
#         packed = compressed["weight_packed"]
#         scale = compressed["weight_scale"]
#         packed_bytes = packed.nelement() * (1 if packed.dtype == torch.uint8 else 4)
#         scale_bytes = scale.nelement() * 4
#         if "weight_global_scale" in compressed:
#             global_scale_bytes = compressed["weight_global_scale"].nelement() * 4
#         else:
#             global_scale_bytes = 0
#         total_compressed = packed_bytes + scale_bytes + global_scale_bytes
#         compression_ratio = (weight.nelement() * 4) / total_compressed

#         results.append({
#             "method": name,
#             "mse": mse,
#             "rel_l2": rel_l2,
#             "cos_sim": cos_sim,
#             "zero_ratio": zero_ratio,
#             "compression": compression_ratio,
#         })

#     print(f"\n{'Method':<24} {'MSE':>10} {'Rel L2':>10} {'Cosine':>10} {'Zero%':>8} {'Compression':>10}")
#     print("-" * 86)
#     for r in results:
#         print(f"{r['method']:<24} {r['mse']:>10.6f} {r['rel_l2']:>10.6f} {r['cos_sim']:>10.6f} {r['zero_ratio']:>8.1f}% {r['compression']:>10.2f}x")

#     # 验证 FP4 打包/解包
#     print(f"\nFP4 打包/解包验证:")
#     test_fp4 = torch.tensor([
#         [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
#         [-0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0],
#     ], dtype=torch.float32)
#     test_packed = pack_fp4_to_uint8(test_fp4)
#     test_unpacked = unpack_fp4_from_uint8(test_packed, test_fp4.shape, dtype=torch.float32)
#     print(f"  输入:   {test_fp4[0].tolist()}")
#     print(f"  打包:   {test_packed[0].tolist()}")
#     print(f"  解包:   {test_unpacked[0].tolist()}")
#     fp4_match = torch.allclose(test_fp4, test_unpacked, atol=1e-6)
#     print(f"  精确还原: {'✅ PASS' if fp4_match else '❌ FAIL'}")

#     print()


# def demo_comparison():
#     """INT4 vs FP4 对比"""
#     print("=" * 70)
#     print("INT4 vs FP4 量化对比")
#     print("=" * 70)

#     torch.manual_seed(42)
#     weight = torch.randn(512, 1024, dtype=torch.float32) * 0.1
#     weight[0, ::64] += 2.0

#     results = []

#     for name, quantizer in [
#         ("INT4 (g=128)", INT4Quantizer(group_size=128, symmetric=True)),
#         ("INT4 (g=64)",  INT4Quantizer(group_size=64, symmetric=True)),
#         ("FP4  (g=16)",  FP4Quantizer(group_size=16)),
#         ("FP4  (g=32)",  FP4Quantizer(group_size=32)),
#     ]:
#         compressed = quantizer.quantize(weight)
#         dequantized = quantizer.dequantize(compressed)
#         error = weight - dequantized
#         mse = torch.mean(error ** 2).item()
#         rel_l2 = (torch.norm(error) / torch.norm(weight)).item()
#         cos_sim = torch.nn.functional.cosine_similarity(
#             weight.flatten().unsqueeze(0),
#             dequantized.flatten().unsqueeze(0)
#         ).item()

#         # 计算实际压缩大小
#         packed = compressed["weight_packed"]
#         scale = compressed["weight_scale"]
#         packed_bytes = packed.nelement() * (4 if packed.dtype == torch.int32 else 1)
#         scale_bytes = scale.nelement() * 4
#         total_bytes = packed_bytes + scale_bytes
#         original_bytes = weight.nelement() * 4

#         results.append({
#             "method": name,
#             "mse": mse,
#             "rel_l2": rel_l2,
#             "cos_sim": cos_sim,
#             "compression": original_bytes / total_bytes,
#         })

#     print(f"\n{'Method':<16} {'MSE':>12} {'Rel L2':>10} {'Cosine':>10} {'Compression':>12}")
#     print("-" * 62)
#     for r in results:
#         print(f"{r['method']:<16} {r['mse']:>12.8f} {r['rel_l2']:>10.6f} {r['cos_sim']:>10.6f} {r['compression']:>10.2f}x")

#     print()


# if __name__ == "__main__":
#     print("  INT4 / FP4 量化器复刻实现 — 源自 compressed-tensors 仓库")
#     print("  仓库: https://github.com/vllm-project/compressed-tensors")

#     # demo_int4()
#     # demo_fp4()
#     # demo_comparison()

#     demo_fp4_no_global_scale()

#     print("=" * 70)
#     print("所有演示完成!")
#     print("=" * 70)
# sure_quant

基于 PyTorch 的旋转量化（Rotation Quantization）框架。通过在量化前对特征空间进行可学习的正交旋转（Hadamard + Givens），使低比特量化的精度损失显著降低。

## 核心思想

传统的均匀量化在大维度特征空间中效果不佳，因为不同维度的数值分布差异很大。sure_quant 通过以下 pipeline 解决该问题：

```
x [N, D]
  │
  ▼ blockify(block_size=g)      # 划分为 M = D/g 个块
x_blk [N, M, g]
  │
  ▼ CompositeBlockRotation      # Hadamard (固定) + Givens (可学习)
z [N, M, g]
  │
  ▼ BlockUniformQuantizer       # 均匀量化 + Straight-Through Estimator
z_hat [N, M, g]
  │
  ▼ CompositeBlockRotation.inverse()
x_hat_blk [N, M, g]
  │
  ▼ deblockify()
x_hat [N, D]                    # 量化后重建的向量
```

## 目录结构

```
sure_quant/
├── ops/                      # 底层运算符
│   ├── block_ops.py          # blockify / deblockify
│   ├── hadamard.py           # 块级 Hadamard 变换 + FWHT
│   └── givens.py             # 可学习 Givens 旋转
├── model/                    # 模型组件
│   ├── wrappers.py           # CompositeBlockRotation（组合旋转）
│   ├── sure_quantizer.py     # SureQuantizer（完整量化器）
│   └── rotated_linear.py     # RotatedQuantLinear（用于替换 nn.Linear）
├── quant/                    # 量化器
│   └── fake_quant.py         # BlockUniformQuantizer（STE 均匀量化）
├── loss/                     # 损失函数
│   ├── reconstruction.py     # 重建损失（MSE）
│   ├── dkoleo.py             # D Kolmogorov 能量损失（促进均匀分布）
│   ├── balance.py            # 平衡损失（零均值、单位方差）
│   ├── range_loss.py         # 范围损失（限制动态范围）
│   └── total_loss.py         # 总损失构造器
├── train/                    # 校准/训练
│   ├── calibrate_rotations.py  # 单层校准
│   └── high_level_api.py     # RotationQuantCalibrator（高层 API）
├── export/                   # 模型序列化
│   ├── export_rotation_params.py  # 导出量化参数
│   └── checkpoint_io.py      # 从文件加载
├── config/                   # 配置
│   └── default_config.py     # RotationQuantConfig（默认配置）
├── scripts/                  # 运行脚本
│   └── run_single_layer_calibration.py
├── tests/                    # 单元测试（pytest）
├── conftest.py               # pytest 路径配置
├── requirements.txt          # Python 依赖
└── README.md                 # 本文件
```

## 环境要求

- Python >= 3.9
- PyTorch >= 2.0.0
- pytest >= 7.0.0（仅测试时需要）

## 安装

```bash
# 1. 克隆项目
cd /path/to/your/workspace
git clone <your-repo-url> sure_quant
cd sure_quant

# 2. 安装依赖
pip install -r requirements.txt

# 3. 在父目录运行代码（包名为 sure_quant，目录名需为 sure_quant）
# 或将 sure_quant 的父目录加入 PYTHONPATH
export PYTHONPATH=/path/to/parent:$PYTHONPATH
```

## 快速上手

### 1. 单层校准

```bash
cd /path/to/parent
python -m sure_quant.scripts.run_single_layer_calibration \
    --dim 4096 \
    --block-size 16 \
    --num-bits 4 \
    --steps 500 \
    --lr 1e-2 \
    --lambda-dk 0.05 \
    --output layer_sure_quant.pt
```

### 2. Python API 使用

```python
import torch
from sure_quant.config.default_config import RotationQuantConfig
from sure_quant.model.sure_quantizer import SureQuantizer
from sure_quant.train.calibrate_rotations import calibrate_single_layer
from sure_quant.train.high_level_api import RotationQuantCalibrator
from sure_quant.export.export_rotation_params import export_sure_quantizer
from sure_quant.export.checkpoint_io import load_sure_quantizer

# ------ 方式一：直接使用 SureQuantizer ------
cfg = RotationQuantConfig()
rq = SureQuantizer(
    dim=4096,
    block_size=cfg.block_size,      # 16
    num_bits=cfg.num_bits,          # 4
    order=cfg.order,                # "hadamard_givens"
)

# 生成校准数据（实际使用中替换为真实激活值）
x = torch.randn(2048, 4096)

# 校准
logs = calibrate_single_layer(rq, x, cfg)

# 推理
out = rq(x)
x_recon = out["x_hat"]              # 重建后的向量
z = out["z"]                         # 旋转空间中间结果

# ------ 方式二：使用高层 API RotationQuantCalibrator ------
calibrator = RotationQuantCalibrator(cfg)
calibrator.calibrate("layer_1", x)
export_sure_quantizer(calibrator.layer_quantizers["layer_1"], "layer_1.pt")

# ------ 方式三：替换已有模型中的 nn.Linear ------
from sure_quant.model.rotated_linear import RotatedQuantLinear

# 将一个已训练的 nn.Linear 替换为带旋转量化的线性层
trained_linear = torch.nn.Linear(512, 256)
quant_linear = RotatedQuantLinear(trained_linear, rq)   # 使用已校准的 rq

# 前向传播
y = quant_linear(torch.randn(32, 512))
```

### 3. 配置说明

`RotationQuantConfig` 的关键字段：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `block_size` | 16 | 特征分块大小 g，D 必须能被 g 整除 |
| `num_bits` | 4 | 量化比特数 |
| `num_givens_layers` | 2 | Givens 旋转层数 |
| `num_pairs_per_layer` | 8 | 每层 Givens 旋转的配对数 |
| `lambda_rec` | 1.0 | 重建损失权重 |
| `lambda_dk` | 0.05 | DKoleo 损失权重 |
| `lambda_bal` | 0.01 | 平衡损失权重 |
| `lambda_range` | 0.01 | 范围损失权重 |
| `calibration_steps` | 500 | 校准步数 |
| `calibration_lr` | 1e-2 | 校准学习率 |
| `device` | "cuda" | 运行设备，自动回退到 cpu |

## 单元测试

```bash
cd /path/to/sure_quant
python -m pytest tests/ -v
```

测试覆盖：

- `test_block_ops.py` — blockify/deblockify 形状与值一致性
- `test_hadamard.py` — Hadamard 变换正逆一致性、正交性、FWHT
- `test_givens.py` — Givens 旋转可学习性、正交性、配对构建
- `test_quant.py` — 均匀量化器 STE 梯度、比特数边界、形状
- `test_dkoleo.py` — DKoleo 损失统计性质与梯度
- `test_pipeline.py` — 端到端 pipeline、训练、导入导出、损失函数集成

## 设计要点

1. **Hadamard 固定，Givens 可学习**：Hadamard 提供零成本的强基变换，Givens 通过少量参数（每层几对角度）微调旋转，参数量远低于直接学习一个正交矩阵。
2. **Straight-Through Estimator (STE)**：量化本身不可导，使用 STE 将梯度直通传递，使整个 pipeline 端到端可训练。
3. **多目标损失**：重建（MSE）+ 分布正则化（DKoleo、balance、range），在保证精度的同时让量化后的分布更均匀。
4. **块级处理**：所有操作按块并行，内存和计算复杂度均为 O(N·D)，与输入大小线性相关。

## 许可证

MIT License

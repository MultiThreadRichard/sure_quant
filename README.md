# sure_quant

基于 PyTorch 的旋转量化框架，支持两种可插拔策略：
- `rotation`：Hadamard + Givens
- `stiefel`：基于stiefel流形约束的Householder（k 反射子）

项目目标是在低比特量化下，同时优化重建误差（MSE）与分布质量（如 KL）。

---

## 核心流程

```text
x [N, D]
  -> blockify
x_blk [N, M, g]
  -> rotation strategy (rotation / stiefel)
z [N, M, g]
  -> BlockUniformQuantizer (STE)
z_hat [N, M, g]
  -> inverse rotation
x_hat_blk [N, M, g]
  -> deblockify
x_hat [N, D]
```

其中 `M = D / g`。

---

## 目录结构

```text
sure_quant/
├── config/
│   └── default_config.py          # SureQuantConfig
├── ops/
│   ├── block_ops.py               # blockify / deblockify
│   ├── hadamard.py                # Hadamard + FWHT
│   └── givens.py                  # Givens 旋转
├── quant/
│   └── fake_quant.py              # BlockUniformQuantizer (STE)
├── model/
│   ├── wrappers.py                # CompositeBlockRotation / StiefelHouseholderRotation
│   ├── sure_quantizer.py          # SureQuantizer（策略注入）
│   └── sure_quant_linear.py       # SureQuantLinear（推理包装）
├── loss/
│   ├── reconstruction.py
│   ├── dkoleo.py
│   ├── balance.py
│   ├── range_loss.py
│   ├── total_loss.py
│   └── joint_objective.py         # stiefel 联合目标
├── train/
│   ├── calibrate_rotations.py     # calibrate_rotation
│   ├── calibrate_stiefel.py
│   ├── stiefel_optimizer.py
│   └── high_level_api.py          # SureQuantCalibrator
├── export/
│   ├── export_rotation_params.py
│   └── checkpoint_io.py
├── scripts/
│   └── run_single_layer_calibration.py
├── tests/
├── requirements.txt
└── README.md
```

---

## 环境要求

- Python >= 3.9
- torch==2.11.0
- pytest==7.4.4

安装：

```bash
pip install -r requirements.txt
```

---

## 快速开始

### 1) CLI 单层校准

```bash
python scripts/run_single_layer_calibration.py \
  --dim 4096 \
  --block-size 16 \
  --num-bits 4 \
  --steps 500 \
  --lr 1e-2 \
  --lambda-dk 0.05 \
  --output layer_sure_quant.pt
```

### 2) Python API（rotation）

```python
import torch
from config.default_config import SureQuantConfig
from model.sure_quantizer import SureQuantizer
from train.calibrate_rotations import calibrate_rotation

cfg = SureQuantConfig()
rq = SureQuantizer(
    dim=4096,
    block_size=cfg.block_size,
    num_bits=cfg.num_bits,
    order=cfg.order,
    rotation_strategy="rotation",
)

x = torch.randn(2048, 4096)
logs = calibrate_rotation(rq, x, cfg)
out = rq(x)
print(out["x_hat"].shape)
```

### 3) Python API（stiefel）

```python
import torch
from config.default_config import SureQuantConfig
from train.calibrate_stiefel import calibrate_stiefel

cfg = SureQuantConfig()
cfg.stiefel_num_reflectors = 8
x = torch.randn(2048, 4096)

result = calibrate_stiefel(x, cfg)
rq_stiefel = result["quantizer"]
out = rq_stiefel(x)
print(out["x_hat"].shape)
```

### 4) 推理包装线性层

```python
import torch
from model.sure_quant_linear import SureQuantLinear

linear = torch.nn.Linear(512, 256)
wrapped = SureQuantLinear(linear, rq_stiefel)
y = wrapped(torch.randn(32, 512))
```

---

## 配置说明（SureQuantConfig）

关键字段（`config/default_config.py`）：

- 量化：`block_size`, `num_bits`
- rotation：`givens_pairs_strategy`, `num_givens_layers`, `num_pairs_per_layer`, `order`
- 损失权重：`lambda_rec`, `lambda_dk`, `lambda_bal`, `lambda_range`
- 训练：`calibration_steps`, `calibration_lr`, `calibration_batch_size`, `dk_sample_size`
- stiefel：`stiefel_num_reflectors`
- 设备：`device`, `dtype`

---

## 测试

```bash
python -m pytest tests/ -v
```

当前测试覆盖：
- 基础算子：`block/hadamard/givens/quant/dkoleo`
- 训练与策略：`calibrate_stiefel`, `stiefel_joint`
- 端到端：`test_pipeline.py`（含 rotation vs stiefel 的 MSE / KL / 网格调参对比）

---

## 设计要点

1. **策略模式**：`SureQuantizer` 支持 `rotation` 与 `stiefel` 可插拔注入。  
2. **STE 可导量化**：前向离散、反向近似恒等，保证训练可行。  
3. **DKoleo 作为关键正则**：除重建外，优化旋转后分布的均匀性与量化友好性。  
4. **可导出/可加载**：导出按策略保存参数，加载自动重建对应量化器。

---

## License

MIT

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from config.default_config import SureQuantConfig
from loss.reconstruction import kl_reconstruction_loss
from model.sure_quant_linear import SureQuantLinear
from model.sure_quantizer import SureQuantizer


def calibrate_weight_rotation(
    quantizer: SureQuantizer,
    weight: torch.Tensor,
    cfg: SureQuantConfig,
) -> list[dict[str, float | int]]:
    """Legacy weight-only calibration helper retained for API compatibility."""
    device = weight.device
    quantizer.to(device).train()
    optimizer = torch.optim.AdamW(quantizer.rotation.parameters(), lr=cfg.calibration_lr)
    values = weight.detach().T.contiguous()
    logs = []
    for step in range(cfg.calibration_steps):
        if len(values) > cfg.calibration_batch_size:
            indices = torch.randperm(len(values), device=device)[: cfg.calibration_batch_size]
            batch = values[indices]
        else:
            batch = values
        output = quantizer(batch)
        loss = kl_reconstruction_loss(
            output["z"],
            output["z_hat"],
            temperature=cfg.kl_temperature,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        logs.append({"step": step, "loss_rec": loss_value, "loss_kl": loss_value})
        del output, loss, batch
    quantizer.eval()
    return logs


def calibrate_all_quantizers(
    model: nn.Module,
    calibration_data: dict[str, torch.Tensor],
    cfg: SureQuantConfig,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Calibrate activation and weight rotations, then bake weights."""
    from train.calibrate_rotations import calibrate_rotation

    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, SureQuantLinear)
    ]
    logs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for index, (name, module) in enumerate(modules, start=1):
        print(f"Calibrating {name} ({index}/{len(modules)})")
        device = module.linear.weight.device
        layer_logs: dict[str, list[dict[str, Any]]] = {}
        if name in calibration_data:
            values = calibration_data[name]
            if values.shape[-1] != module.activation_quantizer.dim:
                raise ValueError(
                    f"{name}: activation dimension {values.shape[-1]} does not match "
                    f"{module.activation_quantizer.dim}"
                )
            layer_logs["activation"] = calibrate_rotation(
                module.activation_quantizer.to(device), values.to(device), cfg
            )
        else:
            print(f"No activation data for {name}; skipping activation calibration")

        if module.weight_quantizer is not None:
            layer_logs["weight"] = calibrate_rotation(
                module.weight_quantizer.to(device),
                module.linear.weight.detach().T.contiguous(),
                cfg,
            )
            module.quantize_weight()
        logs[name] = layer_logs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return logs


@torch.inference_mode()
def reconstruction_score(
    model: nn.Module,
    validation_data: dict[str, torch.Tensor],
    *,
    batch_size: int,
    temperature: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Calculate held-out block-distribution KL divergence."""
    layer_scores: dict[str, float] = {}
    for name, module in model.named_modules():
        if not isinstance(module, SureQuantLinear) or name not in validation_data:
            continue
        values = validation_data[name]
        divergence_sum = 0.0
        block_count = 0
        device = module.linear.weight.device
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size].to(device)
            output = module.activation_quantizer(batch)
            loss = kl_reconstruction_loss(
                output["z"],
                output["z_hat"],
                temperature=temperature,
            )
            current_blocks = output["z"].shape[0] * output["z"].shape[1]
            divergence_sum += float(loss) * current_blocks
            block_count += current_blocks
        if block_count:
            layer_scores[name] = divergence_sum / block_count
    if not layer_scores:
        raise RuntimeError("No quantized layer matched the validation activations")
    return sum(layer_scores.values()) / len(layer_scores), layer_scores

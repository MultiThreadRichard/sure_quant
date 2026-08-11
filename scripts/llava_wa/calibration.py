from __future__ import annotations

from typing import Any

import torch
from torch import nn

from config.default_config import SureQuantConfig
from loss.reconstruction import reconstruction_loss
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
        loss = reconstruction_loss(output["x_blk"], output["x_hat_blk"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logs.append({"step": step, "loss_rec": float(loss.detach())})
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
            # The weight quantizer is only needed during calibration.  Once the
            # quantized weight has been baked into ``module.linear.weight``,
            # release its parameters so they do not consume GPU memory while
            # the next layers are being processed.
            # module.quantize_weight() only execute once
            module.weight_quantizer.to("cpu")
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
) -> tuple[float, dict[str, float]]:
    """Calculate held-out activation reconstruction MSE."""
    layer_scores: dict[str, float] = {}
    for name, module in model.named_modules():
        if not isinstance(module, SureQuantLinear) or name not in validation_data:
            continue
        values = validation_data[name]
        squared_error = 0.0
        element_count = 0
        device = module.linear.weight.device
        for start in range(0, len(values), batch_size):
            output = module.activation_quantizer(values[start : start + batch_size].to(device))
            difference = output["x_blk"] - output["x_hat_blk"]
            squared_error += float(difference.float().square().sum())
            element_count += difference.numel()
        if element_count:
            layer_scores[name] = squared_error / element_count
    if not layer_scores:
        raise RuntimeError("No quantized layer matched the validation activations")
    return sum(layer_scores.values()) / len(layer_scores), layer_scores


@torch.inference_mode()
def reconstruction_score_logits_mse(
    model: nn.Module,
    eval_pairs: list[tuple[dict[str, torch.Tensor], torch.Tensor]],
    *,
    max_new_tokens: int = 128,
) -> float:
    mse_error = 0.0
    pair_count = len(eval_pairs)
    dev = model.device
    for pair in eval_pairs:
        inputs, output_ids = pair
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        output_ids = output_ids.to(dev)

        output_ids_hat = model.generate(**inputs, max_new_tokens=max_new_tokens)[0]
        print(f"output_ids.shape: {output_ids.shape}, output_ids_hat.shape: {output_ids_hat.shape}")

        min_len = min(output_ids.shape[-1], output_ids_hat.shape[-1])
        ids_ref = output_ids[..., :min_len].to(torch.float32)
        ids_hat = output_ids_hat[..., :min_len].to(torch.float32)
 
        err = (ids_ref - ids_hat).pow(2).mean().item()
        mse_error += err
        print(f"mse_error: {mse_error}")

        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        output_ids = output_ids.to("cpu")
        break
    return mse_error / pair_count


# @torch.inference_mode()
# def reconstruction_score_logits_kl(
#     model: nn.Module,
#     eval_pairs: list[tuple[dict[str, torch.Tensor], torch.Tensor]],
#     *,
#     max_new_tokens: int = 128,
# ) -> float:
#     kl_error = 0.0
#     pair_count = len(eval_pairs)
#     dev = model.device
#     for pair in eval_pairs:
#         inputs, output_ids = pair
#         inputs = {k: v.to(dev) for k, v in inputs.items()}
#         output_ids = output_ids.to(dev)

#         output_ids_hat = model.generate(**inputs, max_new_tokens=max_new_tokens)[0]
#         # print(f"output_ids.shape: {output_ids.shape}, output_ids_hat.shape: {output_ids_hat.shape}")

#         kl = compute_kl_for_quantization(output_ids, output_ids_hat)
 
#         kl_error += kl
#         print(f"kl_error: {kl_error}")

#         inputs = {k: v.to("cpu") for k, v in inputs.items()}
#         output_ids = output_ids.to("cpu")
#         break
#     return kl_error / pair_count


@torch.inference_mode()
def reconstruction_score_logits_kl(
    model: nn.Module,
    eval_pairs: list[tuple[dict[str, torch.Tensor], torch.Tensor]],
    *,
    max_new_tokens: int = 128,
    kl_weight: float = 0.5,
    length_weight: float = 0.5,
    penalty_factor: int = 10,
) -> tuple[float, list[dict[str, float]]]:
    """Composite evaluation score combining KL divergence and length mismatch.

    The score is ``kl_weight * norm_kl + length_weight * penalty_factor * norm_length``
    """
    # kl_error = 0.0
    # length_penalty = 0.0
    pair_count = len(eval_pairs)
    dev = model.device
    final_score = 0.0
    detail_scores = []

    for pair in eval_pairs:
        inputs, output_ids = pair

        # 移动到 GPU 并立即使用
        inputs_gpu = {k: v.to(dev) for k, v in inputs.items()}

        # 执行生成（峰值显存在此处不可避免）
        output_ids_hat_gpu = model.generate(**inputs_gpu, max_new_tokens=max_new_tokens)[0]

        #【关键】生成后立即将结果移回 CPU，并删除 GPU 引用
        output_ids_hat_cpu = output_ids_hat_gpu.cpu()
        output_ids_cpu = output_ids.cpu()

        out_len_ref = output_ids_cpu.shape[-1] - inputs_gpu["input_ids"].shape[-1]
        out_len_hat = output_ids_hat_cpu.shape[-1] - inputs_gpu["input_ids"].shape[-1]
        del inputs_gpu, output_ids_hat_gpu  # 显式释放 GPU 张量引用

        # 1) KL 分量
        norm_kl = compute_kl_for_quantization(output_ids_cpu, output_ids_hat_cpu)

        # 2) 长度惩罚：相对差值
        # len_ref = output_ids_cpu.shape[-1]
        # len_hat = output_ids_hat_cpu.shape[-1]
        # norm_length = float(abs(len_hat - len_ref) / len_ref)

        norm_length = float(abs(out_len_hat - out_len_ref) / out_len_ref)
        # print(f"[reconstruction_score_logits_kl] out_len_ref: {out_len_ref}, out_len_hat: {out_len_hat}, norm_length: {norm_length}")

        score = kl_weight * norm_kl + length_weight * penalty_factor * norm_length
        final_score += score
        detail_scores.append({"norm_kl": norm_kl, "norm_length": norm_length, "score": score})

        # kl_error += norm_kl
        # length_penalty += norm_length

    # avg_kl = kl_error / pair_count
    # avg_length = length_penalty / pair_count
    # final_score = kl_weight * avg_kl + length_weight * avg_length

    return final_score / pair_count, detail_scores


def compute_kl_for_quantization(
    fp_logits: torch.Tensor,  # original output logits
    q_logits: torch.Tensor,   # quantized output logits
    bins: int = 256,          # 直方图分箱数(bin)
    eps: float = 1e-10,        # 防止 log(0)
) -> float:
    # 1. 展平
    fp_flat = fp_logits.to(torch.float32).detach().cpu().flatten()
    q_flat = q_logits.to(torch.float32).detach().cpu().flatten()

    # 2. 统一取值范围（必须用相同的 min/max 分箱，否则 KL 无意义）
    min_val = min(fp_flat.min(), q_flat.min())
    max_val = max(fp_flat.max(), q_flat.max())

    # 3. 把一个一维张量里的数字，分成若干区间，统计每个区间有多少个数，返回每个区间的数量
    fp_hist = torch.histc(fp_flat, bins=bins, min=min_val, max=max_val)
    q_hist = torch.histc(q_flat, bins=bins, min=min_val, max=max_val)

    # 4. 转化为频率分布，norm到[0,1] 防止后续KL计算发生nan
    p = fp_hist / (fp_hist.sum())
    q = q_hist / (q_hist.sum())

    # 5. 数值安全保护
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)

    # 6. 计算 KL(P || Q)：用量化分布 Q 近似真实分布 P
    kl = torch.sum(p * torch.log(p / q))

    return kl.item()


# @torch.inference_mode()
# def last_layer_reconstruction_score(
#     model: nn.Module,
#     validation_data: dict[str, torch.Tensor],
#     *,
#     batch_size: int,
# ) -> tuple[float, dict[str, float]]:
#     """Calculate held-out reconstruction MSE for the *last* quantized layer only.

#     Instead of averaging MSE across every ``SureQuantLinear`` layer, this
#     function identifies the final layer in the model (in the order returned by
#     ``model.named_modules``) and computes its reconstruction MSE only.

#     Returns:
#         A tuple ``(score, layer_scores)`` where ``score`` equals the MSE of the
#         last layer and ``layer_scores`` contains that single entry, mirroring
#         the API of :func:`reconstruction_score`.
#     """
#     # Collect every quantized layer that has a matching validation tensor.
#     matched: list[tuple[str, SureQuantLinear]] = []
#     for name, module in model.named_modules():
#         if isinstance(module, SureQuantLinear) and name in validation_data:
#             matched.append((name, module))
#     if not matched:
#         raise RuntimeError("No quantized layer matched the validation activations")

#     last_name, last_module = matched[-1]
#     values = validation_data[last_name]
#     squared_error = 0.0
#     element_count = 0
#     device = last_module.linear.weight.device
#     for start in range(0, len(values), batch_size):
#         output = last_module.activation_quantizer(values[start : start + batch_size].to(device))
#         difference = output["x_blk"] - output["x_hat_blk"]
#         squared_error += float(difference.float().square().sum())
#         element_count += difference.numel()

#     if element_count == 0:
#         raise RuntimeError(f"Last quantized layer '{last_name}' produced no elements")

#     score = squared_error / element_count
#     return score, {last_name: score}
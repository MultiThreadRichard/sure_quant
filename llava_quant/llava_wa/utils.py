from __future__ import annotations

import torch
import torch.nn.functional as F



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


def compute_step_kl(
    fp_step_logits,
    q_step_logits,
) -> tuple[float, int]:
    """逐 decode step 的 next-token 分布 KL(fp || q)，再对步数取平均。

    Args:
        fp_step_logits: 全精度模型每步的 logits，``[vocab]`` / ``[1, vocab]`` 的
            列表（或直接是 ``[T, vocab]`` 张量，按行迭代）。
        q_step_logits:  量化模型同一步的 logits，格式同上。

    Returns:
        ``(mean_kl, n_compared_steps)``。两边提前 EOS 时只比较都产生过的
        ``min(len_fp, len_q)`` 步；无重叠步时返回 ``(nan, 0)``。
    """
    n = min(len(fp_step_logits), len(q_step_logits))
    if n == 0:
        return float("nan"), 0
    kl_sum = 0.0
    for i in range(n):
        # log_target=True 且 input=q：sum(exp(p) * (p - q)) == KL(p || q)
        fp_logp = F.log_softmax(fp_step_logits[i].float(), dim=-1)
        q_logp = F.log_softmax(q_step_logits[i].float(), dim=-1)
        kl_sum += F.kl_div(q_logp, fp_logp, reduction="sum", log_target=True).item()
    return kl_sum / n, n


def compute_cos_similarity(fp_weight: torch.Tensor, q_weight: torch.Tensor):
    fp_weight = fp_weight.detach().cpu().flatten().float()
    q_weight = q_weight.detach().cpu().flatten().float()
    eval_len = min(len(fp_weight), len(q_weight))
    fp_weight = fp_weight[:eval_len]
    q_weight = q_weight[:eval_len]
    return F.cosine_similarity(fp_weight, q_weight, dim=0).item()


def compute_pearson_correlation(x: torch.Tensor, y: torch.Tensor):
    """
    计算两个张量的皮尔逊相关系数 PCC
    x, y: 任意形状的张量（会自动展平）
    返回: PCC 值，范围 [-1,1]
    """
    # 展平成一维
    x = x.detach().cpu().flatten().float()
    y = y.detach().cpu().flatten().float()

    eval_len = min(len(x), len(y))
    x = x[:eval_len]
    y = y[:eval_len]

    # 减去均值
    x_mean = x - x.mean()
    y_mean = y - y.mean()

    # 计算分子（协方差部分）
    numerator = (x_mean * y_mean).sum()
    
    # 计算分母（标准差乘积）
    denominator = torch.sqrt(torch.sum(x_mean ** 2)) * torch.sqrt(torch.sum(y_mean ** 2))
    
    # 防止除 0
    eps = 1e-8
    pcc = numerator / (denominator + eps)

    return pcc.item()


# ---------------------------------------------------------------------------
# vs full-precision baseline: shared greedy decode loop
# ---------------------------------------------------------------------------
# Every quantized-vs-full-precision experiment needs the two sides to run
# through the *same* decoding path, otherwise the comparison mixes quantization
# effects with decoding differences.  ``decode_greedy`` takes an optional
# ``quantizer``; the full-precision baseline is simply the same call with
# ``quantizer=None``.  Running one loop for both is also what makes the
# per-step next-token logits available to ``compute_step_kl`` above.


def _decode_with_native_update(quantizer, past_kv, seq_len_before: int):
    """Call the method-specific decode quantizer (keeps the native cache in sync)."""
    if hasattr(quantizer, "quantize_decode_with_native_update"):  # SureQuant
        return quantizer.quantize_decode_with_native_update(past_kv, seq_len_before)
    return quantizer.quantize_decode_with_native_kv_update(past_kv, seq_len_before)  # TurboQuant


def _release_native_cache(*quantizers) -> None:
    """Drop the quantizers' stashed native KV caches.

    Both quantizers keep a full-precision copy of the cache for the MSE metric
    (SureQuant as GPU tensors, TurboQuant as host numpy arrays).  For a 2k-token
    sample that is ~1 GiB each, so it must not survive the sample that produced
    it.  Safe to clear: both ``quantize_prefill`` implementations re-create it,
    and the decode entry points only run after a prefill.
    """
    for quantizer in quantizers:
        quantizer.native_past_kv = None


@torch.no_grad()
def decode_greedy(
    model,
    inputs,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    quantizer=None,
) -> dict:
    """Greedy prefill/decode loop, optionally quantizing the KV cache each step.

    <for both original transformers model and quantized model>

    The full-precision baseline runs through this exact same loop with
    ``quantizer=None``, so both sides share the decoding path and both expose the
    per-step next-token logits required by the step-wise KL.  Quantized decode
    steps use the native-updating variant so each quantizer retains a full native
    cache for the MSE metric.

    Args:
        model: LLaVA model in eval mode.
        inputs: processor outputs (``input_ids`` / ``attention_mask`` /
            ``pixel_values``) on the model's device.
        max_new_tokens: hard cap on decode steps; generation also stops early
            as soon as ``eos_token_id`` is produced.
        eos_token_id: stop token; ``None`` disables early stopping.
        quantizer: KV-cache quantizer exposing ``quantize_prefill`` plus one of
            the decode-update entry points, or ``None`` for full precision.

    Returns:
        ``{"generated": ids [1, T], "past_kv": final cache,
           "step_logits": one float32 ``[1, vocab]`` CPU tensor per decode step}``.
    """
    from transformers import DynamicCache

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    past_kv = DynamicCache()
    generated = input_ids
    step_logits: list[torch.Tensor] = []

    for step in range(max_new_tokens):
        if step == 0:
            outputs = model(
                input_ids=generated,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = outputs.past_key_values
            if quantizer is not None:
                past_kv = quantizer.quantize_prefill(past_kv)
        else:
            seq_len_before = generated.shape[1] - 1
            outputs = model(
                input_ids=generated[:, -1:],
                attention_mask=attention_mask,
                pixel_values=None,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = outputs.past_key_values
            if quantizer is not None:
                past_kv = _decode_with_native_update(quantizer, past_kv, seq_len_before)

        next_logits = outputs.logits[:, -1, :]
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        step_logits.append(next_logits.float().cpu())
        generated = torch.cat([generated, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

        del outputs

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return {"generated": generated, "past_kv": past_kv, "step_logits": step_logits}


@torch.no_grad()
def generate_step_logits_for_original(
    model,
    inputs,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    **generate_kwargs,
) -> dict:
    """Per-step next-token logits from ``generate`` for original transformers model.

    Same ``step_logits`` contract as ``decode_greedy`` above, but collected from
    HuggingFace's generation loop instead of a hand-written one.  ``generate``
    records the *raw* pre-softmax logits under ``output_logits=True``
    (``GenerationMixin._sample`` keeps ``outputs.logits[:, -1, :].clone().float()``
    separately from the post-``logits_processor`` ``output_scores``), so no
    quantizer hook and no manual prefill/decode bookkeeping are needed here.

    This is NOT interchangeable with ``decode_greedy`` for a quantized run: the
    default path has nowhere to inject a KV-cache quantizer.  It exists for the
    full-precision side and for callers that only want the logits.

    Note: ``generate`` forces ``num_logits_to_keep=1``, so the LM head only sees
    the last hidden state while ``decode_greedy`` computes logits for every
    position and slices.  The two differ by ~1e-2 in fp16 -- same order as a
    quantization effect -- so pick one capture path and use it for both sides of
    a comparison.

    Args:
        model: LLaVA model in eval mode.
        inputs: processor outputs (``input_ids`` / ``attention_mask`` /
            ``pixel_values``) on the model's device.
        max_new_tokens: hard cap on decode steps.
        eos_token_id: forwarded to ``generate`` only when not ``None``; left
            unset so the model's generation config decides, otherwise passing
            ``None`` would disable early stopping.
        **generate_kwargs: forwarded to ``generate``.  Must not activate a
            ``logits_processor`` (``repetition_penalty`` / ``min_length`` /
            ``suppress_tokens`` / ``renormalize_logits`` / warpers ...): those
            rewrite the scores the sampler argmaxes over, so the tokens returned
            in ``generated`` would no longer be the argmax of ``step_logits``.

    Returns:
        ``{"generated": ids [1, prompt_len + T],
           "step_logits": one float32 ``[1, vocab]`` CPU tensor per decode step}``.
    """
    if eos_token_id is not None:
        generate_kwargs["eos_token_id"] = eos_token_id

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        output_logits=True,
        return_dict_in_generate=True,
        **generate_kwargs,
    )

    return {
        "generated": outputs.sequences,
        "step_logits": [logits.float().cpu() for logits in outputs.logits],
    }
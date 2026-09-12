import torch, torch.nn as nn, torch.nn.functional as F, argparse, datetime
from transformers import AutoProcessor, LlavaForConditionalGeneration, DynamicCache
from PIL import Image
from typing import Tuple
import numpy as np

import os
import time
from datasets import load_dataset
from tqdm import tqdm
from qwen_vl_utils import process_vision_info


import sys
from pathlib import Path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant import TurboQuant, TurboQuantMSE, KVCacheCompressor
from turboquant.outlier import OutlierTurboQuant



def turbo_compress_kv(kv: dict):
    """Compress real KV tensors and measure quality at various bit-widths."""
    # print("\n" + "=" * 70)
    # print("COMPRESSION QUALITY ON REAL KV TENSORS")
    # print("=" * 70)

    k_cache = kv["k_cache"]
    v_cache = kv["v_cache"]
    num_layers, num_heads, seq_len, head_dim = k_cache.shape

    # print(f"\n  Model KV shape: {k_cache.shape}")
    # print(f"  Total vectors: {num_layers * num_heads * seq_len}")
    # print(f"  Original size: {k_cache.nbytes + v_cache.nbytes:,} bytes "
    #       f"({(k_cache.nbytes + v_cache.nbytes) / 1024 / 1024:.1f} MB)")

    # print(f"\n  {'Config':<22} {'K MSE':>12} {'V MSE':>12} {'K Cosine':>10} {'V Cosine':>10} {'Ratio':>8}")
    # print(f"  {'─' * 80}")

    configs = [
        # ("Uniform 2-bit", 2, 2, "uniform"),
        # ("Outlier 2.5-bit", 2.5, 2.5, "outlier"),
        # ("Uniform 3-bit", 3, 3, "uniform"),
        # ("Outlier 3.5-bit", 3.5, 3.5, "outlier"),
        ("Uniform 4-bit", 4, 4, "uniform"),
    ]

    new_kv = {}
    for name, k_bits, v_bits, mode in configs:
        if mode == "uniform":
            compressor = KVCacheCompressor(head_dim=head_dim, k_bits=int(k_bits), v_bits=int(v_bits))
            compressed = compressor.compress(k_cache, v_cache)
            k_hat, v_hat = compressor.decompress(compressed)
            stats = compressor.memory_stats(seq_len, num_layers, num_heads)
            ratio = stats["compression_ratio"]
        else:
            # Outlier: compress each head individually
            k_hat, v_hat, ratio = _compress_outlier(k_cache, v_cache, k_bits, v_bits, head_dim)

        # k_mse = np.mean((k_cache - k_hat) ** 2)
        # v_mse = np.mean((v_cache - v_hat) ** 2)

        # # Per-vector cosine similarity
        # k_flat = k_cache.reshape(-1, head_dim)
        # k_hat_flat = k_hat.reshape(-1, head_dim)
        # cosines = _batch_cosine_sim(k_flat, k_hat_flat)

        # v_flat = v_cache.reshape(-1, head_dim)
        # v_hat_flat = v_hat.reshape(-1, head_dim)
        # v_cosines = _batch_cosine_sim(v_flat, v_hat_flat)

        # print(f"  {name:<22} {k_mse:>12.8f} {v_mse:>12.8f} {np.mean(cosines):>10.6f} {np.mean(v_cosines):>10.6f} {ratio:>7.1f}×")
        
        new_kv["k_cache"] = k_hat
        new_kv["v_cache"] = v_hat
    
    return new_kv, ratio


def _batch_cosine_sim(A, B):
    """Cosine similarity between corresponding rows."""
    dots = np.sum(A * B, axis=1)
    norms_a = np.linalg.norm(A, axis=1)
    norms_b = np.linalg.norm(B, axis=1)
    valid = (norms_a > 1e-10) & (norms_b > 1e-10)
    cos = np.zeros(len(A))
    cos[valid] = dots[valid] / (norms_a[valid] * norms_b[valid])
    return cos


def _compress_outlier(k_cache, v_cache, k_bits, v_bits, head_dim):
    """Compress with outlier strategy, per-head."""
    num_layers, num_heads, seq_len, _ = k_cache.shape
    k_hat = np.zeros_like(k_cache)
    v_hat = np.zeros_like(v_cache)

    for layer in range(num_layers):
        for head in range(num_heads):
            # K cache with outlier TurboQuant
            k_oq = OutlierTurboQuant(head_dim, target_bits=k_bits, seed=42 + layer * 100 + head)
            k_vecs = k_cache[layer, head]
            for i in range(seq_len):
                c = k_oq.quantize(k_vecs[i])
                k_hat[layer, head, i] = k_oq.dequantize(c)

            # V cache with outlier PolarQuant (MSE-only, lower overhead)
            v_oq = OutlierTurboQuant(head_dim, target_bits=v_bits, seed=42 + layer * 100 + head + 50)
            v_vecs = v_cache[layer, head]
            for i in range(seq_len):
                c = v_oq.quantize(v_vecs[i])
                v_hat[layer, head, i] = v_oq.dequantize(c)

    # Approximate ratio, original fp16 for one coordinate of head dim
    avg_bits = (k_bits + v_bits) / 2
    ratio = 16 / (avg_bits + 96 / head_dim)  # 32+64 bits for 2 norms per vector
    return k_hat, v_hat, ratio




def llava_full_infer(raw_image):
    checkpoint = "/home/ccwan/stu_Jiangtp/model_repo/llava-7b-hf"

    # original model
    model = LlavaForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float16).eval()
    processor = AutoProcessor.from_pretrained(checkpoint)

    # Confirm generations of the quantized model look sane.
    print("========== SAMPLE GENERATION ==============")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please describe the animal in this image\n"},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128)
    print(processor.decode(output[0], skip_special_tokens=True))
    print("==========================================")
    return output[0]


def eval_out_logits(new_output, raw_image):
    from eval_out_logits import compute_cos_similarity, compute_pearson_correlation, compute_kl_for_quantization
    original_output = llava_full_infer(raw_image)

    print(f"original_output.shape: {original_output.shape}")
    print(f"new_output.shape: {new_output.shape}")

    cos_sim = compute_cos_similarity(original_output, new_output)
    print(f"Cosine Similarity: {cos_sim}")
    pearson_corr = compute_pearson_correlation(original_output, new_output)
    print(f"Pearson Correlation: {pearson_corr}")
    kl_div = compute_kl_for_quantization(original_output, new_output)
    print(f"KL Divergence: {kl_div}")


class LLaVAKVOptimizedQuantizer:
    def __init__(self, model):
        self.model = model
        self.ratio = None
        self.native_past_kv: dict | None = None # 模型原始未量化的kv cache


    
    def extract_kv(self, past_kv: DynamicCache):
        # print(f"len(past_kv): {len(past_kv)}")
        # 批量转换 key_cache value_cache 到 numpy array
        # shape: [num_layers, num_heads, seq_len, head_dim]
        k_cache = np.stack([k.squeeze(0).detach().cpu().numpy() for k in past_kv.key_cache])
        v_cache = np.stack([v.squeeze(0).detach().cpu().numpy() for v in past_kv.value_cache])
        # print(f"k_cache.shape: {k_cache.shape}")
        # print(f"v_cache.shape: {v_cache.shape}")

        return {"k_cache": k_cache, "v_cache": v_cache}
    

    def assign_kv_to_cache(self, past_kv: DynamicCache, k_cache: np.ndarray, v_cache: np.ndarray) -> DynamicCache:
        """
        将 numpy 数组形式的 k_cache 和 v_cache 转换回 PyTorch tensor，并赋值给 DynamicCache
        
        Args:
            past_kv: 目标 DynamicCache 对象
            k_cache: numpy 数组，形状 [num_layers, num_heads, seq_len, head_dim]
            v_cache: numpy 数组，形状 [num_layers, num_heads, seq_len, head_dim]
        
        Returns:
            更新后的 DynamicCache
        """
        num_layers = k_cache.shape[0]
        
        # 确保 k_cache 和 v_cache 形状一致
        assert k_cache.shape == v_cache.shape, "k_cache 和 v_cache 形状必须一致"
        assert num_layers == len(past_kv), f"层数不匹配: k_cache 有 {num_layers} 层, past_kv 有 {len(past_kv)} 层"
        
        # 获取原始 tensor 的设备和数据类型
        device = past_kv.key_cache[0].device
        dtype = past_kv.key_cache[0].dtype
        
        # 逐层转换并赋值
        for i in range(num_layers):
            # numpy → tensor，添加 batch 维度（unsqueeze(0)）
            k_tensor = torch.from_numpy(k_cache[i]).unsqueeze(0).to(device=device, dtype=dtype)
            v_tensor = torch.from_numpy(v_cache[i]).unsqueeze(0).to(device=device, dtype=dtype)
            
            # 更新缓存
            past_kv.key_cache[i] = k_tensor
            past_kv.value_cache[i] = v_tensor
        
        # print(f"成功赋值 {num_layers} 层的 KV 缓存")
        # print(f"第0层 k 形状: {past_kv.key_cache[0].shape}")
        # print(f"第0层 v 形状: {past_kv.value_cache[0].shape}")
        
        return past_kv


    def quantize_prefill(self, past_kv: DynamicCache) -> DynamicCache:
        # print(past_kv.key_cache[0])

        kv = self.extract_kv(past_kv)
        self.native_past_kv = kv # 保存原始未量化的kv cache

        new_kv, self.ratio = turbo_compress_kv(kv)
        past_kv = self.assign_kv_to_cache(past_kv, new_kv["k_cache"], new_kv["v_cache"])

        # print(past_kv.key_cache[0])
        return past_kv
    

    def quantize_decode_with_native_kv_update(self, past_kv: DynamicCache, seq_len_before: int) -> DynamicCache:
        ''' 动态推理decode阶段, 量化kv cache新增的部分, 同时更新 native_past_kv 中的完整未量化kv, 用于后续evaluate_metrics '''
        # 防御性检查
        assert self.native_past_kv is not None, "native_past_kv 未初始化, quantize_decode_incremental之前必须先调用 quantize_prefill"
        
        # 收集所有层的新增KV（用于更新 native_past_kv）
        all_k_new = []
        all_v_new = []

        for i in range(len(past_kv)):
            k, v = past_kv[i]

            # 历史已量化KV
            k_hist = k[:, :, :seq_len_before, :]
            v_hist = v[:, :, :seq_len_before, :]

            # 新增1个token的KV（仅量化这里）
            k_new = k[:, :, seq_len_before:, :]
            v_new = v[:, :, seq_len_before:, :]
            
            # 收集新增KV（仅转换一次）
            all_k_new.append(k_new.detach().cpu())
            all_v_new.append(v_new.detach().cpu())

            # 构造字典并调用 turbo_compress_kv
            k_new_np = k_new.detach().cpu().numpy()
            v_new_np = v_new.detach().cpu().numpy()
            
            kv_dict = {
                "k_cache": k_new_np,
                "v_cache": v_new_np
            }
            new_kv_dict, _ = turbo_compress_kv(kv_dict)

            # 转换回 tensor
            device = k.device
            dtype = k.dtype
            k_new_hat = torch.from_numpy(new_kv_dict["k_cache"]).to(device=device, dtype=dtype)
            v_new_hat = torch.from_numpy(new_kv_dict["v_cache"]).to(device=device, dtype=dtype)
            
            past_kv.key_cache[i] = torch.cat([k_hist, k_new_hat], dim=2)
            past_kv.value_cache[i] = torch.cat([v_hist, v_new_hat], dim=2)

        # 更新 native_past_kv
        k_np = torch.cat(all_k_new, dim=0).numpy()
        v_np = torch.cat(all_v_new, dim=0).numpy()
        self.native_past_kv['k_cache'] = np.concatenate([self.native_past_kv['k_cache'], k_np], axis=2)
        self.native_past_kv['v_cache'] = np.concatenate([self.native_past_kv['v_cache'], v_np], axis=2)

        return past_kv
    

    def quantize_decode_incremental(self, past_kv: DynamicCache, seq_len_before: int) -> DynamicCache:
        ''' 仅动态推理decode阶段, 量化kv cache新增的部分, 不更新 native_past_kv 中的完整未量化kv, 避免拖慢推理速度 '''
        # # save new kv to native_past_kv
        # # print(f"self.native_past_kv['k_cache'].shape: {self.native_past_kv['k_cache'].shape}")
        # # print(f"self.native_past_kv['v_cache'].shape: {self.native_past_kv['v_cache'].shape}")

        # # (32, 32, 1, 128)
        # k_cache_new = torch.cat(past_kv.key_cache, dim=0)[:, :, seq_len_before:, :]
        # v_cache_new = torch.cat(past_kv.value_cache, dim=0)[:, :, seq_len_before:, :]

        # k_np = k_cache_new.detach().cpu().numpy()
        # v_np = v_cache_new.detach().cpu().numpy()

        # self.native_past_kv['k_cache'] = np.concatenate([self.native_past_kv['k_cache'], k_np], axis=2)
        # self.native_past_kv['v_cache'] = np.concatenate([self.native_past_kv['v_cache'], v_np], axis=2)

        # # print(f"self.native_past_kv['k_cache'].shape: {self.native_past_kv['k_cache'].shape}")
        # # print(f"self.native_past_kv['v_cache'].shape: {self.native_past_kv['v_cache'].shape}")

        for i in range(len(past_kv)):
            k, v = past_kv[i]
            # print(f"Layer {i} | k shape: {k.shape}, v shape: {v.shape}")

            # 历史已量化KV
            k_hist = k[:, :, :seq_len_before, :]
            v_hist = v[:, :, :seq_len_before, :]

            # 新增1个token的KV（仅量化这里）
            k_new = k[:, :, seq_len_before:, :]
            v_new = v[:, :, seq_len_before:, :]

            k_new_np = k_new.detach().cpu().numpy()
            v_new_np = v_new.detach().cpu().numpy()
            # print(f"k_new_np.dtype: {k_new_np.dtype}, v_new_np.dtype: {v_new_np.dtype}")
            
            # 构造字典并调用 turbo_compress_kv
            kv_dict = {
                "k_cache": k_new_np,
                "v_cache": v_new_np
            }
            new_kv_dict, _ = turbo_compress_kv(kv_dict)

            # 转换回 tensor
            device = k.device
            dtype = k.dtype

            k_new_hat = torch.from_numpy(new_kv_dict["k_cache"]).to(device=device, dtype=dtype)
            v_new_hat = torch.from_numpy(new_kv_dict["v_cache"]).to(device=device, dtype=dtype)
            
            past_kv.key_cache[i] = torch.cat([k_hist, k_new_hat], dim=2)
            past_kv.value_cache[i] = torch.cat([v_hist, v_new_hat], dim=2)
            
            # print(f"Layer {i} | Dequantized k shape: {past_kv.key_cache[i].shape}, Dequantized v shape: {past_kv.value_cache[i].shape}")
            # print(f"type(past_kv.key_cache[i]): {type(past_kv.key_cache[i])}")
            # print(f"type(past_kv.value_cache[i]): {type(past_kv.value_cache[i])}")
        return past_kv


    def evaluate_metrics(self, past_kv: DynamicCache):
        assert self.ratio is not None, "ratio 未初始化, 请先调用 quantize_prefill 量化kv"
        # before turboquant
        k_cache = self.native_past_kv["k_cache"]
        v_cache = self.native_past_kv["v_cache"]

        # after turboquant
        kv = self.extract_kv(past_kv)
        k_hat = kv["k_cache"]
        v_hat = kv["v_cache"]

        assert k_cache.shape == k_hat.shape, "k_cache.shape != k_hat.shape"
        assert v_cache.shape == v_hat.shape, "v_cache.shape != v_hat.shape"

        head_dim = k_cache.shape[-1]

        k_mse = np.mean((k_cache - k_hat) ** 2)
        v_mse = np.mean((v_cache - v_hat) ** 2)

        # Per-vector cosine similarity
        k_flat = k_cache.reshape(-1, head_dim)
        k_hat_flat = k_hat.reshape(-1, head_dim)
        cosines = _batch_cosine_sim(k_flat, k_hat_flat)

        v_flat = v_cache.reshape(-1, head_dim)
        v_hat_flat = v_hat.reshape(-1, head_dim)
        v_cosines = _batch_cosine_sim(v_flat, v_hat_flat)

        # print(f"\n  {'K MSE':>12} {'V MSE':>12} {'K Cosine':>10} {'V Cosine':>10} {'Ratio':>8}")
        # print(f"  {'─' * 80}")
        # print(f"  {k_mse:>12.8f} {v_mse:>12.8f} {np.mean(cosines):>10.6f} {np.mean(v_cosines):>10.6f} {self.ratio:>7.1f}×")
        
        # 计算 Pearson 相关系数（需要转换为 torch.Tensor）
        k_tensor = torch.from_numpy(k_cache)
        k_hat_tensor = torch.from_numpy(k_hat)
        v_tensor = torch.from_numpy(v_cache)
        v_hat_tensor = torch.from_numpy(v_hat)
        
        # k_pearson = compute_pearson_correlation(k_tensor, k_hat_tensor)
        # v_pearson = compute_pearson_correlation(v_tensor, v_hat_tensor)
 
        # # 计算 KL 散度
        # k_kl = compute_kl_for_quantization(k_tensor, k_hat_tensor)
        # v_kl = compute_kl_for_quantization(v_tensor, v_hat_tensor)
 
        print(f"\n  {'K MSE':>12} {'V MSE':>12} {'K Cosine':>10} {'V Cosine':>10} {'Ratio':>8}")
        print(f"  {'─' * 100}")
        print(f"  {k_mse:>12.8f} {v_mse:>12.8f} {np.mean(cosines):>10.6f} {np.mean(v_cosines):>10.6f} {self.ratio:>7.1f}×")




# ===================== INT4 RTN 量化器 =====================
class INT4RTNQuantizer:
    INT4_MIN = -8
    INT4_MAX = 7

    @staticmethod
    def quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_val = torch.abs(x).max().clamp(min=1e-8)
        scale = max_val / INT4RTNQuantizer.INT4_MAX
        quantized = torch.round(x / scale).to(torch.int8)
        quantized = torch.clamp(quantized, INT4RTNQuantizer.INT4_MIN, INT4RTNQuantizer.INT4_MAX)
        return quantized, scale

    @staticmethod
    def dequantize(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return quantized.float() * scale



class LLaVAInferEngine:
    def __init__(
        self,
        model,
        processor,
    ):
        self.processor = processor
        self.model = model

        self.kv_quant = LLaVAKVOptimizedQuantizer(self.model)

    @torch.no_grad()
    def generate(self, raw_image, messages, max_new_tokens=128, need_eval = False, temperature=0.1, do_sample=False):
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(images=raw_image, text=prompt, return_tensors="pt").to(self.model.device)
        
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        images = inputs.pixel_values
        print(f"input_ids shape: {input_ids.shape}")

        past_key_values = DynamicCache()
        generated = input_ids
        eos_token_id = self.processor.tokenizer.eos_token_id
        print(f"eos_token_id: {eos_token_id}")

        # need_eval = False
        for step in range(max_new_tokens):
            # ==========================================
            # 当前KV Cache已有长度
            # ==========================================
            # current_seq_len_before = generated.shape[1]
            current_seq_len_before = past_key_values.seen_tokens
            print(f">>>>> current_seq_len_before: {current_seq_len_before}")
            
            if step == 0:
                # Prefill：全量量化
                outputs = self.model(
                    input_ids=generated,
                    attention_mask=attention_mask,
                    pixel_values=images,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                # print(f"type(outputs.past_key_values): {type(outputs.past_key_values)}")
                # print(f"outputs.past_key_values: {outputs.past_key_values}")
                past_key_values = self.kv_quant.quantize_prefill(outputs.past_key_values)

                # break
            else:
                # Decode：增量量化
                outputs = self.model(
                    input_ids=generated[:, -1:],
                    attention_mask=attention_mask,
                    pixel_values=None,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                if need_eval:
                    past_key_values = self.kv_quant.quantize_decode_with_native_kv_update(
                        outputs.past_key_values,
                        seq_len_before=current_seq_len_before
                    )
                else:
                    past_key_values = self.kv_quant.quantize_decode_incremental(
                        outputs.past_key_values,
                        seq_len_before=current_seq_len_before
                    )

            # 采样下一个token
            logits = outputs.logits[:, -1, :] / temperature

            if do_sample:
                # 根据概率分布随机采样 (等价于 transformers generate(do_sample=True))
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            else:
                # 使用 argmax 贪婪采样
                next_token = logits.argmax(dim=-1, keepdim=True)

            print(f"next_token: {next_token}")

            # 追加新token
            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

            print(f">>>>> Step {step} | Tokens len: {generated.shape[1]}")

            if next_token.item() == eos_token_id or generated.shape[1] >= input_ids.shape[1] + max_new_tokens:
                break
        
        # evaluate kv cache metric
        if need_eval:
            self.kv_quant.evaluate_metrics(past_key_values)
        
        # eval_out_logits(generated[0], raw_image)

        return self.processor.decode(generated[0], skip_special_tokens=True)


    @torch.no_grad()
    def generate_for_mme(self, inputs, max_new_tokens=128, need_eval=False, temperature=0.1, do_sample=False):     
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        images = inputs.pixel_values
        # print(f"input_ids shape: {input_ids.shape}")

        past_key_values = DynamicCache()
        generated = input_ids
        eos_token_id = self.processor.tokenizer.eos_token_id
        # print(f"eos_token_id: {eos_token_id}")

        # need_eval = False
        for step in range(max_new_tokens):
            # ==========================================
            # 当前KV Cache已有长度
            # ==========================================
            # current_seq_len_before = generated.shape[1]
            current_seq_len_before = past_key_values.seen_tokens
            # print(f">>>>> current_seq_len_before: {current_seq_len_before}")
            
            if step == 0:
                # Prefill：全量量化
                outputs = self.model(
                    input_ids=generated,
                    attention_mask=attention_mask,
                    pixel_values=images,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                # print(f"type(outputs.past_key_values): {type(outputs.past_key_values)}")
                # print(f"outputs.past_key_values: {outputs.past_key_values}")
                past_key_values = self.kv_quant.quantize_prefill(outputs.past_key_values)

                # break
            else:
                # Decode：增量量化
                outputs = self.model(
                    input_ids=generated[:, -1:],
                    attention_mask=attention_mask,
                    pixel_values=None,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                if need_eval:
                    past_key_values = self.kv_quant.quantize_decode_with_native_kv_update(
                        outputs.past_key_values,
                        seq_len_before=current_seq_len_before
                    )
                else:
                    past_key_values = self.kv_quant.quantize_decode_incremental(
                        outputs.past_key_values,
                        seq_len_before=current_seq_len_before
                    )

            # 采样下一个token
            logits = outputs.logits[:, -1, :] / temperature

            if do_sample:
                # 根据概率分布随机采样 (等价于 transformers generate(do_sample=True))
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            else:
                # 使用 argmax 贪婪采样
                next_token = logits.argmax(dim=-1, keepdim=True)

            # print(f"next_token: {next_token}")

            # 追加新token
            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

            # print(f">>>>> Step {step} | Tokens len: {generated.shape[1]}")

            if next_token.item() == eos_token_id or generated.shape[1] >= input_ids.shape[1] + max_new_tokens:
                break
        
        # evaluate kv cache metric
        if need_eval:
            self.kv_quant.evaluate_metrics(past_key_values)

        return generated


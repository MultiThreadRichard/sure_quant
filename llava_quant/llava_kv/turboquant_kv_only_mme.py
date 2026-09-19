#!/usr/bin/env python3
"""MME evaluation of LLaVA with ONLY the KV cache quantized (TurboQuant).

Scope: weights and activations stay full-precision; the sole quantization is
the decoder's KV cache, compressed by TurboQuant (``KVCacheCompressor``, 4-bit
uniform) inside ``LLaVAInferEngine``.

Contrast with ``sure_wa_turbo_mme.py``, which loads a saved W4A16 SureQuant
model (weights quantized) and additionally runs TurboQuant on the KV cache.

Usage:
CUDA_VISIBLE_DEVICES=0 nohup python -u llava_quant/llava_wa_kv/turboquant_kv_only_mme.py \
    > logs/sure_kv_dev/mme_turboquant_kv_only.log 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from llava_quant.llava_wa.config import CHECKPOINT, PATH_PREFIX


MME_DATA_PATH_LIST = [
    f'{PATH_PREFIX}/workspace/data/MME/data/test-00000-of-00004-a25dbe3b44c4fda6.parquet',
    f'{PATH_PREFIX}/workspace/data/MME/data/test-00001-of-00004-7d22c7f1aba6fca4.parquet',
    f'{PATH_PREFIX}/workspace/data/MME/data/test-00002-of-00004-594798fd3f5b029c.parquet',
    f'{PATH_PREFIX}/workspace/data/MME/data/test-00003-of-00004-53ae1794f93b1e35.parquet',
]

MME_OUTPUT_PATH = (
    f"{PATH_PREFIX}/workspace/sure_quant/logs/mme_eval_res_llava_kv_only_turboquant"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MME eval of LLaVA with ONLY the KV cache quantized (TurboQuant)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-path", type=str, default=MME_OUTPUT_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_dataset_from_local(path: str):
    """Load an MME parquet split into (dataset, chat messages)."""
    trainset = load_dataset('parquet', data_files=path, split='train')
    print(f'len(trainset): {len(trainset)}')

    messages = []
    for item in trainset:
        msg_item = [{
            "role": "user",
            "content": [
                {"type": "image", "image": item['image']},
                {"type": "text", "text": item['question']}
            ]
        }]
        messages.append(msg_item)

    return trainset, messages


def mme_test(engine, processor, args: argparse.Namespace) -> None:
    """Run the full MME set through the TurboQuant KV-quantized engine."""
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    turn = 0
    for data_path in MME_DATA_PATH_LIST:
        t_data, messages = load_dataset_from_local(data_path)
        print(f'>>>>>>>>> load {data_path}')

        print('>>>>>>>>> start eval')
        with open(os.path.join(output_path, f'eval_results0{turn}.txt'),
                  'a', encoding="utf-8") as fout:
            for item, msg_item in tqdm(zip(t_data, messages)):
                text = processor.apply_chat_template(
                    msg_item, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(msg_item)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(args.device)

                generated_ids = engine.generate_for_mme(
                    inputs=inputs,
                    max_new_tokens=args.max_new_tokens,
                )

                response = processor.batch_decode(
                    generated_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                print(item['category'], item['question_id'], item['question'],
                      item['answer'], response, sep='\t', file=fout)

        print('>>>>>>>>> end eval')
        torch.cuda.empty_cache()
        turn += 1

    print(f'>>>>>>>>> mme complete turn: {turn}')


def main() -> None:
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    from mme.llava_kv_quant_turbo import LLaVAInferEngine

    args = parse_args()

    print("=" * 60)
    print("LLaVA MME — KV cache only, TurboQuant")
    print("=" * 60)
    print(f"  Model:      {CHECKPOINT} (full precision, weights/activations untouched)")
    print(f"  KV Cache:   4-bit uniform (TurboQuant)")
    print(f"  Output:     {args.output_path}")
    print("=" * 60)

    # Full-precision model: only the KV cache is quantized.
    model = LlavaForConditionalGeneration.from_pretrained(
        CHECKPOINT, device_map=args.device, torch_dtype=torch.float16,
    ).eval()
    processor = AutoProcessor.from_pretrained(CHECKPOINT)

    engine = LLaVAInferEngine(model, processor)

    mme_test(engine, processor, args)


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f">>>>>>>>>>>>> Done, elapsed time: {elapsed:.2f} seconds")

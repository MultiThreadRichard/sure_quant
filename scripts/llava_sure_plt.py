import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
import gc
import json
import argparse
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image


from scripts.llava_wa.config import (
    PATH_PREFIX,
    DEFAULT_INFERENCE_PROMPT,
    build_parser,
)
from scripts.llava_wa.modeling import (
    quantize_linear_layer,
    quantize_llava_model,
    selected_linear_names,
)
from scripts.llava_wa.calibration import (
    calibrate_all_quantizers,
    reconstruction_score,
)
from scripts.llava_wa.data import (
    collect_calibration_data,
    split_calibration_data,
    make_prompt,
    generate_assistant_outputs,
)
from scripts.llava_wa.persistence import save_quantized_model, load_quantized_model, _jsonable_config
from scripts.llava_wa.search import seed_everything

from scripts.plt_tools import (
    collect_llava_vision_input_activations,
    plt_llava_vision_activation,
    plt_llava_vision_weight,
    plt_act_val_dim,
    plt_act_after_surequant,
    plt_sure_encoder_x2d_xhat,
)


"""
加载后surequant模型, 可视化
TODO: current for int4 quantized model
"""


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
CHECKPOINT = f"{PATH_PREFIX}/workspace/models/llava-1.5-7b-hf"
SAMPLE_IMG_DIR = f"{PATH_PREFIX}/workspace/awq_learn/sample_img"
# CALIB_DATA_PATH = f"{PATH_PREFIX}/workspace/data/flickr30k/data/test-00000-of-00009.parquet"
# SAVE_ID = "02"
# DEFAULT_SAVE_DIR = f"{PATH_PREFIX}/workspace/sure_quant/model_saved/llava_7b_sure_fp4_{SAVE_ID}"




# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def infer(
    model: LlavaForConditionalGeneration,
    processor: Any,
    img_path: str,
    prompt_text: str = DEFAULT_INFERENCE_PROMPT,
    max_new_tokens: int = 128,
) -> torch.Tensor:
    """Run inference on a single image and print the result."""
    print("========== SAMPLE GENERATION ============")
    prompt = make_prompt(processor, prompt_text)
    raw_image = Image.open(img_path)
    device = next(model.parameters()).device

    inputs = processor(
        images=raw_image, text=prompt, return_tensors="pt",
    ).to(device)
    print(f"inputs['input_ids'].shape: {inputs['input_ids'].shape}")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = processor.decode(output[0], skip_special_tokens=True)
    print(f"Generated: {decoded}")
    print("==========================================")
    return output[0]


def run_saved_model_int4() -> None:
    """Load a saved quantized model and run inference."""
    save_path = "/home/ecnu01/sure_quant_models/20260803/best_quantized_model"
    print(f"\n========== Loading saved quantized model from {save_path} ==========")

    loaded_model = load_quantized_model(
        save_path, device_map="cuda", torch_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(save_path)

    infer(loaded_model, processor, os.path.join(SAMPLE_IMG_DIR, "sample1.jpg"))
    infer(loaded_model, processor, os.path.join(SAMPLE_IMG_DIR, "sample2.jpg"))



def analyze_llava_activation_after_surequant(
    model: LlavaForConditionalGeneration,
    processor: Any,
    output_path: str,
    img_path: str,
    prompt_text: str = DEFAULT_INFERENCE_PROMPT,
):
    prompt = make_prompt(processor, prompt_text)
    raw_image = Image.open(img_path)
    device = next(model.parameters()).device

    inputs = processor(
        images=raw_image, text=prompt, return_tensors="pt",
    ).to(device)
    print(f"inputs['input_ids'].shape: {inputs['input_ids'].shape}")


    # paint activations vision
    activations = collect_llava_vision_input_activations(model, inputs)
    print(f"activations len: {len(activations)}")
    print(f"activations[0].shape: {activations[0].shape}")

    plt_llava_vision_activation(activations, output_path=output_path)

    print("==========================================")


def analyze_llava_weight_after_surequant(
    model: LlavaForConditionalGeneration,
    output_path: str,
):
    plt_llava_vision_weight(model, output_path=output_path)


def collect_sure_encoder_x2d_xhat(
    model: "LlavaForConditionalGeneration",
    processor: Any,
    img_path: str,
    prompt_text: str = DEFAULT_INFERENCE_PROMPT,
    encoder_prefixes=None,
) -> dict:
    """
    Collect ``x2d`` and ``x_hat`` from every ``SureQuantLinear`` inside the
    CLIP *vision_tower.vision_model.encoder* (by default), keyed by the
    module's full ``state_dict``-style name.

    Returns
    -------
    dict  {"<module_name>": {"x2d": Tensor, "x_hat": Tensor}, ...}
    """
    from model.sure_quant_linear import SureQuantLinear

    # Which sub-tree of the model to walk. Default: CLIP vision encoder.
    if encoder_prefixes is None:
        encoder_prefixes = ["vision_tower.vision_model.encoder"]

    activations: dict = {}

    def _make_hook(name: str):
        def _hook(mod: SureQuantLinear, inputs, output):
            # inputs[0] is the raw (pre-quant) activation tensor ``x``.
            x = inputs[0].detach()
            input_dtype = x.dtype
            # D = mod.linear.in_features
            x2d = x.reshape(-1, x.shape[-1]).contiguous()

            # Reproduce x_hat exactly as forward() does.
            with torch.no_grad():
                x_hat = mod.activation_quantizer(x2d)["x_hat"].detach().to(input_dtype)

            activations[name] = {"x2d": x2d, "x_hat": x_hat}
        return _hook

    # Register hooks on every SureQuantLinear whose name lives under the
    # requested encoder prefix(es).
    handles = []
    for name, mod in model.named_modules():
        if not isinstance(mod, SureQuantLinear):
            continue
        print(f"name: {name}")
        if any(name.startswith(p) for p in encoder_prefixes):
            handles.append(mod.register_forward_hook(_make_hook(name)))

    print(f"[collect_sure_encoder_x2d_xhat] registered {len(handles)} hooks")

    # Run inference to trigger the hooks.
    prompt = make_prompt(processor, prompt_text)
    raw_image = Image.open(img_path)
    device = next(model.parameters()).device
    inputs = processor(
        images=raw_image, text=prompt, return_tensors="pt",
    ).to(device)

    model.eval()
    with torch.no_grad():
        _ = model(**inputs)

    # Clean up hooks so they never leak into subsequent runs.
    for h in handles:
        h.remove()

    print(f"[collect_sure_encoder_x2d_xhat] collected {len(activations)} entries")
    return activations


def collect_sure_encoder_actquant_return(
    model: "LlavaForConditionalGeneration",
    processor: Any,
    img_path: str,
    prompt_text: str = DEFAULT_INFERENCE_PROMPT,
    encoder_prefixes=None,
) -> dict:
    """Collect ``x2d`` plus the **full dict** returned by
    ``mod.activation_quantizer(x2d)`` for every ``SureQuantLinear`` inside the
    CLIP *vision_tower.vision_model.encoder* (by default).

    The returned quantizer dict contains (see :class:`SureQuantizer.forward`):

    ========  =============================================
    key       meaning
    ========  =============================================
    x_blk     blockified input, shape [N, num_blocks, block_size]
    z         rotated input  (rotation(x_blk))
    z_hat     fake-quantized rotated value
    x_hat_blk inverse-rotated reconstruction (still blockified)
    x_hat     deblockified reconstruction, shape [N, dim]
    scale     per-block quantization scale
    ========  =============================================

    Returns
    -------
    dict  {"<module_name>": {"x2d": Tensor, "quant_out": dict}, ...}
    """
    from model.sure_quant_linear import SureQuantLinear

    if encoder_prefixes is None:
        encoder_prefixes = ["vision_tower.vision_model.encoder"]

    activations: dict = {}

    def _make_hook(name: str):
        def _hook(mod: SureQuantLinear, inputs, output):
            # inputs[0] is the raw (pre-quant) activation tensor ``x``.
            x = inputs[0].detach()
            x2d = x.reshape(-1, x.shape[-1]).contiguous()

            # Capture the **entire** quantizer return dict (not only x_hat).
            with torch.no_grad():
                quant_out = {
                    k: v.detach() if isinstance(v, torch.Tensor) else v
                    for k, v in mod.activation_quantizer(x2d).items()
                }

            quant_out.update({"x2d": x2d})
            activations[name] = quant_out
        return _hook

    handles = []
    for name, mod in model.named_modules():
        if not isinstance(mod, SureQuantLinear):
            continue
        if any(name.startswith(p) for p in encoder_prefixes):
            handles.append(mod.register_forward_hook(_make_hook(name)))

    print(f"[collect_sure_encoder_actquant_return] registered {len(handles)} hooks")

    # Run inference to trigger the hooks.
    prompt = make_prompt(processor, prompt_text)
    raw_image = Image.open(img_path)
    device = next(model.parameters()).device
    inputs = processor(
        images=raw_image, text=prompt, return_tensors="pt",
    ).to(device)

    model.eval()
    with torch.no_grad():
        _ = model(**inputs)

    for h in handles:
        h.remove()

    print(f"[collect_sure_encoder_actquant_return] collected {len(activations)} entries")
    return activations


def collect_encoder_attn_mlp_input_activations(
    model: "LlavaForConditionalGeneration",
    processor: Any,
    img_path: str,
    prompt_text: str = DEFAULT_INFERENCE_PROMPT,
) -> dict:
    """Collect input activations of ``self_attn`` and ``mlp`` inside every
    CLIP vision encoder layer, keyed by the module's full ``state_dict``-style
    name.

    Returns
    -------
    dict  {
        "vision_tower.vision_model.encoder.layers.0.self_attn": Tensor[tokens, dim],
        "vision_tower.vision_model.encoder.layers.0.mlp":       Tensor[tokens, dim],
        ...
    }
    """
    activations: dict = {}

    # def _make_hook(name: str):
    #     def _hook(mod, inputs, output):
    #         # inputs[0] is the main input to self_attn / mlp, shape [B, S, H]
    #         print(type(inputs))
    #         print(f"inputs: {inputs}")
    #         inp = inputs[0].detach().cpu().float()
    #         _, _, h = inp.shape
    #         activations[name] = inp.reshape(-1, h)
    #     return _hook
    
    def _make_hook(name: str):
        def _hook(mod, args, kwargs, output):   # 新增 kwargs 参数
            # 获取输入张量
            if args and len(args) > 0:
                inp = args[0]
            else:
                inp = kwargs.get('hidden_states')
                if inp is None:
                    # 可选的 fallback，如果命名不同可扩展
                    raise ValueError(f"Cannot find input tensor for {name}")
            inp = inp.detach().cpu().float()
            _, _, h = inp.shape
            activations[name] = inp.reshape(-1, h)
        return _hook


    encoder_layers = model.vision_tower.vision_model.encoder.layers
    handles = []
    for idx, layer in enumerate(encoder_layers):
        for sub_name in ("self_attn", "mlp"):
            sub_mod = getattr(layer, sub_name)
            full_name = f"vision_tower.vision_model.encoder.layers.{idx}.{sub_name}"
            # 启用关键字参数注册
            handles.append(sub_mod.register_forward_hook(_make_hook(full_name), with_kwargs=True))

    print(f"[collect_encoder_attn_mlp_input_activations] registered {len(handles)} hooks")

    # Run inference to trigger the hooks.
    prompt = make_prompt(processor, prompt_text)
    raw_image = Image.open(img_path)
    device = next(model.parameters()).device
    inputs = processor(
        images=raw_image, text=prompt, return_tensors="pt",
    ).to(device)

    model.eval()
    with torch.no_grad():
        _ = model(**inputs)

    for h in handles:
        h.remove()

    print(f"[collect_encoder_attn_mlp_input_activations] collected {len(activations)} entries")
    return activations


def analyze_sure_llava_clip_encoder_bak(loaded_model: LlavaForConditionalGeneration, processor: AutoProcessor):
    img_path = os.path.join(SAMPLE_IMG_DIR, "sample1.jpg")
    acts = collect_sure_encoder_x2d_xhat(loaded_model, processor, img_path)

    # Inspect a few keys
    # for k in list(acts)[:3]:
    #     print(k, "x2d:", acts[k]["x2d"].shape, "x_hat:", acts[k]["x_hat"].shape)

    # Plot x2d / x_hat
    out_dir = "/home/ecnu01/workspace/sure_quant/logs/sure_rot_qt_vis_before"
    os.makedirs(out_dir, exist_ok=True)
    plt_sure_encoder_x2d_xhat(acts, output_path=out_dir)

    # ---- Collect self_attn / mlp input activations ----
    # img_path = os.path.join(SAMPLE_IMG_DIR, "sample1.jpg")
    # attn_mlp_acts = collect_encoder_attn_mlp_input_activations(
    #     loaded_model, processor, img_path,
    # )

    # # Inspect a few keys
    # # for k in list(attn_mlp_acts):
    # #     print(k, "shape:", attn_mlp_acts[k].shape)

    # # Plot per-layer input activations (one figure per key).
    # # attn_mlp_out_dir = "/home/ecnu01/workspace/sure_quant/logs/sure_attn_mlp_input_vis"
    # # for key, tensor in attn_mlp_acts.items():
    # #     safe_name = key.replace(".", "_")
    # #     plt_act_val_dim(
    # #         tensor.cpu().float().numpy(),
    # #         os.path.join(attn_mlp_out_dir, f"{safe_name}_sure.png"),
    # #         title=f"{key} after SureQuant",
    # #         xlabel="Hidden dimension index",
    # #         ylabel="Activation value",
    # #     )
    
    # attn_mlp_out_dir = "/home/ecnu01/workspace/sure_quant/logs/sure_attn_mlp_input_vis_before"
    # os.makedirs(attn_mlp_out_dir, exist_ok=True)
    # for key, tensor in attn_mlp_acts.items():
    #     safe_name = key.replace(".", "_")
    #     plt_act_val_dim(
    #         tensor.cpu().float().numpy(),
    #         os.path.join(attn_mlp_out_dir, f"{safe_name}_before.png"),
    #         title=f"{key} Before SureQuant",
    #         xlabel="Hidden dimension index",
    #         ylabel="Activation value",
    #     )


def analyze_sure_llava_clip_encoder(loaded_model: LlavaForConditionalGeneration, processor: AutoProcessor):
    img_path = os.path.join(SAMPLE_IMG_DIR, "sample1.jpg")
    acts = collect_sure_encoder_actquant_return(loaded_model, processor, img_path)

    out_dir = "/home/ecnu01/workspace/sure_quant/logs/sure_rot_qt_vis_after"
    os.makedirs(out_dir, exist_ok=True)
    # Inspect a few keys
    for k, v_dict in acts.items():
        # print(f"k: {k}, v: {v_dict.keys()}")
        for name, tensor in v_dict.items():
            # print(f"{name}: {tensor.shape}")

            if name in ["z_hat"]:
                print(f"{name}: {tensor.shape}")
                file_prefix = "_".join(k.split(".")[2:])
                # file_prefix = k.replace(".", "_")

                plt_act_after_surequant(
                    tensor.reshape(tensor.shape[0], -1).cpu().float().numpy(),
                    os.path.join(out_dir, f"{file_prefix}_{name}.png"),
                    title=f"{file_prefix} after SureQuant",
                    xlabel="Hidden dimension index",
                    ylabel="Activation value",
                )

        # break




# def save_sure_encoder_x2d_xhat(activations: dict, save_path: str) -> None:
#     """Save the collected dict to disk (torch.save -> .pt)."""
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     torch.save(activations, save_path)
#     print(f"[save_sure_encoder_x2d_xhat] saved -> {save_path}  "
#           f"(keys={len(activations)})")


# def test_plt():
#     save_path = "/home/ecnu01/sure_quant_models/20260803/best_quantized_model"
#     print(f"\n========== Loading saved quantized model from {save_path} ==========")

#     loaded_model = load_quantized_model(
#         save_path, device_map="cuda", torch_dtype=torch.float16,
#     )

#     processor = AutoProcessor.from_pretrained(save_path)

#     # activation
#     # sure_a_vis_int4_path = "/home/ecnu01/workspace/sure_quant/logs/sure_a_vis_int4"
#     # os.makedirs(sure_a_vis_int4_path, exist_ok=True)

#     # analyze_llava_activation_after_surequant(
#     #     loaded_model,
#     #     processor,
#     #     output_path=sure_a_vis_int4_path,
#     #     img_path=os.path.join(SAMPLE_IMG_DIR, "sample1.jpg")
#     # )

#     # weight
#     sure_w_vis_int4_path = "/home/ecnu01/workspace/sure_quant/logs/sure_w_vis_int4"
#     os.makedirs(sure_w_vis_int4_path, exist_ok=True)

#     analyze_llava_weight_after_surequant(
#         loaded_model,
#         output_path=sure_w_vis_int4_path,
#     )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    # quantized model
    save_path = "/home/ecnu01/sure_quant_models/20260803/best_quantized_model"
    print(f"\n========== Loading saved quantized model from {save_path} ==========")

    loaded_model = load_quantized_model(
        save_path, device_map="cuda", torch_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(save_path)

    # after surequant
    analyze_sure_llava_clip_encoder(loaded_model, processor)

    # before surequant
    # analyze_sure_llava_clip_encoder_bak(loaded_model, processor)


    # original model
    # checkpoint = "/home/ecnu01/workspace/models/llava-1.5-7b-hf"

    # model = LlavaForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float16).eval()
    # processor = AutoProcessor.from_pretrained(checkpoint)

    # analyze_sure_llava_clip_encoder_bak(model, processor)





if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f">>>>>>>>>>>>> Done, elapsed time: {elapsed:.2f} seconds")
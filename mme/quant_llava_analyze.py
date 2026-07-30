import sys
# sys.path.append("/home/ccwan/stu_Jiangtp/origin_quarot/third-party/fast-hadamard-transform")

import os
import time
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

import torch, torch.nn as nn, torch.nn.functional as F, argparse, datetime
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import GenerationConfig
# from transformers import AutoTokenizer, AutoImageProcessor, LlavaProcessor

from qwen_vl_utils import process_vision_info


from fake_quant import quant_utils
from fake_quant import utils
from fake_quant import hadamard_utils

from llava_new import LLaVA
from fake_quant.llava_rotation import fuse_llava_layer_norms, rotate_llava_model, rotate_vision_pre_layernorm
from llava_weight_quant_utils import llava_weight_quant_fwrd_plus
# from llava_kv_quant_turbo import LLaVAInferEngine

from plt_tools import *


torch.set_grad_enabled(False)


"""可视化分析script

CUDA_VISIBLE_DEVICES=0 python mme/quant_llava_analyze.py \
--model_name /home/ecnu01/workspace/models/llava-1.5-7b-hf \
--rotate \
--rotate_visual_clip \
--rotate_visual_cross_attn \
--rotate_llm \
--visual_w_bits 8 \
--visual_a_bits 8 \
--llm_w_bits 8 \
--llm_a_bits 8 \
--quant \
--quant_llm \
--quant_visual_clip \
--quant_cross_attention \
--visual_w_clip \
--llm_w_clip \
--online_llm_hadamard \
--act_order \
--online_visual_hadamard \
--visual_w_rtn \
--llm_w_rtn \
--w_asym


CUDA_VISIBLE_DEVICES=0 python mme/quant_llava_analyze.py \
--model_name /home/ecnu01/workspace/models/llava-1.5-7b-hf \
--rotate \
--rotate_visual_clip \
--rotate_visual_cross_attn \
--rotate_llm \
--visual_w_bits 4 \
--visual_a_bits 4 \
--llm_w_bits 4 \
--llm_a_bits 4 \
--quant \
--quant_llm \
--quant_visual_clip \
--quant_cross_attention \
--visual_w_clip \
--llm_w_clip \
--online_llm_hadamard \
--act_order \
--online_visual_hadamard \
--visual_w_rtn \
--llm_w_rtn \
--w_asym

"""


def llava_full_infer():
    checkpoint = "/home/ccwan/stu_Jiangtp/model_repo/llava-7b-hf"

    # original model
    model = LlavaForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float16).eval()
    processor = AutoProcessor.from_pretrained(checkpoint)

    print(model)


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
    raw_image = Image.open("/home/ccwan/stu_Jiangtp/MQuant/assert/sample1.jpg")


    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128)
        # output = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
    print(processor.decode(output[0], skip_special_tokens=True))
    print("==========================================")


def infer(vlm_llava):
    model = vlm_llava.model
    processor = vlm_llava.processor

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
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "A cat is in the image. Please answer yes or no."},
    #             {"type": "image"},
    #         ],
    #     },
    # ]
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "Two dogs are in the image. Please answer yes or no."},
    #             {"type": "image"},
    #         ],
    #     },
    # ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    # raw_image = Image.open("/home/ccwan/stu_Jiangtp/MQuant/assert/sample1.jpg")
    raw_image = Image.open("/home/ecnu01/workspace/awq_learn/sample_img/sample1.jpg")
    # raw_image = Image.open("/home/ecnu01/workspace/awq_learn/sample_img/sample2.jpg")


    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)
    print(inputs.keys())
    print(f"inputs['input_ids'].shape: {inputs['input_ids'].shape}")


    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128)
        # output = model.generate(
        #     **inputs, 
        #     max_new_tokens=128,
        #     temperature=0.7,
        #     top_p=0.9,
        #     do_sample=True,
        #     num_beams=1,
        #     repetition_penalty=1.1
        # )

    print("output: ", output)
    print("output.shape: ", output.shape)
    
    print(processor.decode(output[0], skip_special_tokens=True))
    print("==========================================")
    return output[0]


def main_w_analyze(args):
    model_name = args.model_name
    model = LLaVA(
        model_path=model_name, verbose=args.verbose
    )

    # print(model_name)
    # print(model.model_path)
    # print(model.model)
    # print(model.model.config)
    # print(model.model.language_model.config)
    # vision_config = model.model.vision_tower.config
    # num_heads = vision_config.num_attention_heads
    # head_dim = vision_config.hidden_size // num_heads
    # print(f"num_heads: {num_heads}")
    # print(f"head_dim: {head_dim}")
    

    utils.seed_everything(args.seed)
    # original_output = infer(model)
    
    if not args.not_fuse_layer_norms:
        fuse_llava_layer_norms(model, args)
        # print(model.model.language_model.model.norm.weight.data)
        # print(model.model.language_model.model.norm.weight.shape)
    # infer(model)


    # TODO 2
    handle_list = []

    if args.rotate:
        Q_v = rotate_llava_model(model.model, args)
        print("rotate Q_v.shape: ", Q_v.shape)

        dev = model.model.vision_tower.vision_model.pre_layrnorm.weight.device
        h = rotate_vision_pre_layernorm(model.model.vision_tower.vision_model.pre_layrnorm, Q_v, dev)
        handle_list.append(h)
    
    print(f"model.model.language_model.config.intermediate_size: {model.model.language_model.config.intermediate_size}")
    print(f"model.model.config.need_pad: {model.model.config.need_pad}")
    # infer(model)


    # # TODO 3
    if args.quant:
        if args.online_llm_hadamard:
            if args.rotate_llm:
                args.quant_llm = True
        if args.online_visual_hadamard:
            if args.rotate_visual_clip:
                args.quant_visual_clip = True
        quant_utils.llava_add_act_qaunt(model, args)

        if args.online_llm_hadamard and args.rotate_llm:
            print("adding online llm hadamard rotation")
            qlayers = quant_utils.find_qlayers(
                model.model.language_model, layers=[quant_utils.ActQuantWrapper]
            )
            
            for name in qlayers:
                # if 'self_attn.o_proj' in name:
                #     had_K, K = hadamard_utils.get_hadK(model.model.language_model.config.num_attention_heads)
                #     qlayers[name].online_partial_had = True
                #     qlayers[name].had_K = had_K
                #     qlayers[name].K = K
                #     qlayers[name].had_dim = model.model.language_model.config.hidden_size // model.model.language_model.config.num_attention_heads
                #     qlayers[name].fp32_had = args.fp32_had
                    
                if "mlp.down_proj" in name:
                    had_K, K = hadamard_utils.get_hadK(
                        model.model.language_model.config.intermediate_size
                    )
                    qlayers[name].online_full_had = True
                    # print(qlayers[name].online_full_had)
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = args.fp32_had
                    # qlayers[name].split = args.llm_split
                    # if args.llm_split:
                    #     qlayers[name].split_weights()
                    # if model.model.config.need_pad:
                    #     hook = functools.partial(
                    #         utils.revise_down_input,
                    #         new_size=model.model.config.intermediate_size,
                    #     )
                    #     qlayers[name].register_forward_pre_hook(hook)

        # print(model.model.language_model.model.layers[0].mlp.down_proj.online_full_had)
        # print(model.model.language_model.model.layers[0].mlp.down_proj.had_K)
        # print(model.model.language_model.model.layers[0].mlp.down_proj.K)


        if args.online_visual_hadamard and args.rotate_visual_clip:
            print("adding online visual hadamard rotation")
            qlayers = quant_utils.find_qlayers(
                model.model.vision_tower, layers=[quant_utils.ActQuantWrapper]
            )
            qlayers_mm = quant_utils.find_qlayers(
                model.model.multi_modal_projector, layers=[quant_utils.ActQuantWrapper]
            )
            qlayers.update(qlayers_mm)

            hsize = int(model.model.vision_tower.vision_model.encoder.layers[0].mlp.fc2.module.in_features)
            for name in qlayers:
                # if "layers.0.self_attn" in name:
                #     if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                #         had_K, K = hadamard_utils.get_hadK(
                #             int(model.model.vision_tower.vision_model.encoder.layers[0].self_attn.q_proj.module.in_features)
                #         )
                #         qlayers[name].online_full_had = True
                #         qlayers[name].had_K = had_K
                #         qlayers[name].K = K
                #         qlayers[name].fp32_had = args.fp32_had
                #         # qlayers[name].split = args.visual_split
                #         # if args.visual_split:
                #         #     qlayers[name].split_weights()

                if "mlp.fc2" in name:
                    had_K, K = hadamard_utils.get_hadK(hsize)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = args.fp32_had
                    # qlayers[name].split = args.visual_split
                    # if args.visual_split:
                    #     qlayers[name].split_weights()


        # TODO weight quant
        quantizers = llava_weight_quant_fwrd_plus(
            model, None, model.model.device, None, args
        )
        print(f">>>>>>>>>>>> weight quant done")


        if args.visual_a_bits < 16 or args.visual_static:
            print(">>>>>>>> visual activation quant configure")
            if args.visual_static and args.visual_a_bits >= 16:
                print("if you want to run act with fp16, please set --static False")
            # qlayers = quant_utils.find_qlayers(
            #     model.model.visual, layers=[quant_utils.ActQuantWrapper]
            # )

            qlayers = quant_utils.find_qlayers(
                model.model.vision_tower, layers=[quant_utils.ActQuantWrapper]
            )
            qlayers_mm = quant_utils.find_qlayers(
                model.model.multi_modal_projector, layers=[quant_utils.ActQuantWrapper]
            )
            qlayers.update(qlayers_mm)

            for name in qlayers:
                if any(p_name in name for p_name in args.skip_names):
                    continue
                layer_input_bits = args.visual_a_bits
                layer_groupsize = args.a_groupsize
                layer_a_sym = not (args.a_asym)
                layer_a_clip = args.a_clip_ratio

                qlayers[name].quantizer.configure(
                    bits=layer_input_bits,
                    groupsize=layer_groupsize,
                    sym=layer_a_sym,
                    clip_ratio=layer_a_clip,
                    act_per_tensor=args.act_per_tensor,
                    static=args.visual_static,
                    observer_type="minmax",
                )

        if args.llm_a_bits < 16 or args.llm_static:
            print(">>>>>>>> llm activation quant configure")
            if args.llm_static and args.llm_a_bits >= 16:
                print("if you want to run act with fp16, please set --static False")
            # qlayers = quant_utils.find_qlayers(
            #     model.model.model, layers=[quant_utils.ActQuantWrapper]
            # )
            qlayers = quant_utils.find_qlayers(
                model.model.language_model, layers=[quant_utils.ActQuantWrapper]
            )
            for name in qlayers:
                if any(p_name in name for p_name in args.skip_names):
                    continue
                layer_input_bits = args.llm_a_bits
                layer_groupsize = args.a_groupsize
                layer_a_sym = not (args.a_asym)
                layer_a_clip = args.a_clip_ratio

                qlayers[name].quantizer.configure(
                    bits=layer_input_bits,
                    groupsize=layer_groupsize,
                    sym=layer_a_sym,
                    clip_ratio=layer_a_clip,
                    act_per_tensor=args.act_per_tensor,
                    static=args.llm_static,
                    observer_type="minmax",
                )


   

    analyze_llava_weight_after_rot(model)

    print(">>>>>>>> done")



def main_a_analyze(args):
    model_name = args.model_name
    model = LLaVA(
        model_path=model_name, verbose=args.verbose
    )

    # print(model_name)
    # print(model.model_path)
    # print(model.model)
    # print(model.model.config)
    # print(model.model.language_model.config)
    # vision_config = model.model.vision_tower.config
    # num_heads = vision_config.num_attention_heads
    # head_dim = vision_config.hidden_size // num_heads
    # print(f"num_heads: {num_heads}")
    # print(f"head_dim: {head_dim}")
    

    utils.seed_everything(args.seed)
    # original_output = infer(model)
    
    if not args.not_fuse_layer_norms:
        fuse_llava_layer_norms(model, args)
        # print(model.model.language_model.model.norm.weight.data)
        # print(model.model.language_model.model.norm.weight.shape)
    # infer(model)


    # TODO 2
    handle_list = []

    if args.rotate:
        Q_v = rotate_llava_model(model.model, args)
        print("rotate Q_v.shape: ", Q_v.shape)

        dev = model.model.vision_tower.vision_model.pre_layrnorm.weight.device
        h = rotate_vision_pre_layernorm(model.model.vision_tower.vision_model.pre_layrnorm, Q_v, dev)
        handle_list.append(h)
    
    print(f"model.model.language_model.config.intermediate_size: {model.model.language_model.config.intermediate_size}")
    print(f"model.model.config.need_pad: {model.model.config.need_pad}")
    # infer(model)


    # # TODO 3
    if args.quant:
        if args.online_llm_hadamard:
            if args.rotate_llm:
                args.quant_llm = True
        if args.online_visual_hadamard:
            if args.rotate_visual_clip:
                args.quant_visual_clip = True
        quant_utils.llava_add_act_qaunt(model, args)

        if args.online_llm_hadamard and args.rotate_llm:
            print("adding online llm hadamard rotation")
            qlayers = quant_utils.find_qlayers(
                model.model.language_model, layers=[quant_utils.ActQuantWrapper]
            )
            
            for name in qlayers:
                # if 'self_attn.o_proj' in name:
                #     had_K, K = hadamard_utils.get_hadK(model.model.language_model.config.num_attention_heads)
                #     qlayers[name].online_partial_had = True
                #     qlayers[name].had_K = had_K
                #     qlayers[name].K = K
                #     qlayers[name].had_dim = model.model.language_model.config.hidden_size // model.model.language_model.config.num_attention_heads
                #     qlayers[name].fp32_had = args.fp32_had
                    
                if "mlp.down_proj" in name:
                    had_K, K = hadamard_utils.get_hadK(
                        model.model.language_model.config.intermediate_size
                    )
                    qlayers[name].online_full_had = True
                    # print(qlayers[name].online_full_had)
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = args.fp32_had
                    # qlayers[name].split = args.llm_split
                    # if args.llm_split:
                    #     qlayers[name].split_weights()
                    # if model.model.config.need_pad:
                    #     hook = functools.partial(
                    #         utils.revise_down_input,
                    #         new_size=model.model.config.intermediate_size,
                    #     )
                    #     qlayers[name].register_forward_pre_hook(hook)

        # print(model.model.language_model.model.layers[0].mlp.down_proj.online_full_had)
        # print(model.model.language_model.model.layers[0].mlp.down_proj.had_K)
        # print(model.model.language_model.model.layers[0].mlp.down_proj.K)


        if args.online_visual_hadamard and args.rotate_visual_clip:
            print("adding online visual hadamard rotation")
            qlayers = quant_utils.find_qlayers(
                model.model.vision_tower, layers=[quant_utils.ActQuantWrapper]
            )
            qlayers_mm = quant_utils.find_qlayers(
                model.model.multi_modal_projector, layers=[quant_utils.ActQuantWrapper]
            )
            qlayers.update(qlayers_mm)

            hsize = int(model.model.vision_tower.vision_model.encoder.layers[0].mlp.fc2.module.in_features)
            for name in qlayers:
                # if "layers.0.self_attn" in name:
                #     if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                #         had_K, K = hadamard_utils.get_hadK(
                #             int(model.model.vision_tower.vision_model.encoder.layers[0].self_attn.q_proj.module.in_features)
                #         )
                #         qlayers[name].online_full_had = True
                #         qlayers[name].had_K = had_K
                #         qlayers[name].K = K
                #         qlayers[name].fp32_had = args.fp32_had
                #         # qlayers[name].split = args.visual_split
                #         # if args.visual_split:
                #         #     qlayers[name].split_weights()

                if "mlp.fc2" in name:
                    had_K, K = hadamard_utils.get_hadK(hsize)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = args.fp32_had
                    # qlayers[name].split = args.visual_split
                    # if args.visual_split:
                    #     qlayers[name].split_weights()


        # TODO weight quant
        quantizers = llava_weight_quant_fwrd_plus(
            model, None, model.model.device, None, args
        )
        print(f">>>>>>>>>>>> weight quant done")


        if args.visual_a_bits < 16 or args.visual_static:
            print(">>>>>>>> visual activation quant configure")
            if args.visual_static and args.visual_a_bits >= 16:
                print("if you want to run act with fp16, please set --static False")
            # qlayers = quant_utils.find_qlayers(
            #     model.model.visual, layers=[quant_utils.ActQuantWrapper]
            # )

            qlayers = quant_utils.find_qlayers(
                model.model.vision_tower, layers=[quant_utils.ActQuantWrapper]
            )
            qlayers_mm = quant_utils.find_qlayers(
                model.model.multi_modal_projector, layers=[quant_utils.ActQuantWrapper]
            )
            qlayers.update(qlayers_mm)

            for name in qlayers:
                if any(p_name in name for p_name in args.skip_names):
                    continue
                layer_input_bits = args.visual_a_bits
                layer_groupsize = args.a_groupsize
                layer_a_sym = not (args.a_asym)
                layer_a_clip = args.a_clip_ratio

                qlayers[name].quantizer.configure(
                    bits=layer_input_bits,
                    groupsize=layer_groupsize,
                    sym=layer_a_sym,
                    clip_ratio=layer_a_clip,
                    act_per_tensor=args.act_per_tensor,
                    static=args.visual_static,
                    observer_type="minmax",
                )

        if args.llm_a_bits < 16 or args.llm_static:
            print(">>>>>>>> llm activation quant configure")
            if args.llm_static and args.llm_a_bits >= 16:
                print("if you want to run act with fp16, please set --static False")
            # qlayers = quant_utils.find_qlayers(
            #     model.model.model, layers=[quant_utils.ActQuantWrapper]
            # )
            qlayers = quant_utils.find_qlayers(
                model.model.language_model, layers=[quant_utils.ActQuantWrapper]
            )
            for name in qlayers:
                if any(p_name in name for p_name in args.skip_names):
                    continue
                layer_input_bits = args.llm_a_bits
                layer_groupsize = args.a_groupsize
                layer_a_sym = not (args.a_asym)
                layer_a_clip = args.a_clip_ratio

                qlayers[name].quantizer.configure(
                    bits=layer_input_bits,
                    groupsize=layer_groupsize,
                    sym=layer_a_sym,
                    clip_ratio=layer_a_clip,
                    act_per_tensor=args.act_per_tensor,
                    static=args.llm_static,
                    observer_type="minmax",
                )



    analyze_llava_activation_after_rot(model)

    print(">>>>>>>> done")


def analyze_llava_activation():
    # checkpoint = "/home/ccwan/stu_Jiangtp/model_repo/llava-7b-hf"
    checkpoint = "/home/ecnu01/workspace/models/llava-1.5-7b-hf"


    # original model
    model = LlavaForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float16).eval()
    processor = AutoProcessor.from_pretrained(checkpoint)

    # print(model)

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
    # raw_image = Image.open("/home/ccwan/stu_Jiangtp/MQuant/assert/sample1.jpg")
    # raw_image = Image.open("/home/ecnu01/workspace/awq_learn/sample_img/sample1.jpg")
    raw_image = Image.open("/home/ecnu01/workspace/awq_learn/sample_img/sample2.jpg")


    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)

    # paint activations llm
    # activations = collect_llava_lang_decoder_input_activations(model, inputs)
    # print(f"activations len: {len(activations)}")
    # print(f"activations[0].shape: {activations[0].shape}")

    # before_rot_lang_path = "/home/ccwan/stu_Jiangtp/MQuant/figs/a_before_rot_w8a8"
    # os.makedirs(before_rot_lang_path, exist_ok=True)
    # plt_llava_lang_activation(activations, output_path=before_rot_lang_path)

    # paint activations vision
    activations = collect_llava_vision_input_activations(model, inputs)
    print(f"activations len: {len(activations)}")
    print(f"activations[0].shape: {activations[0].shape}")

    # before_rot_vis_path = "/home/ccwan/stu_Jiangtp/MQuant/figs/a_vis_before_rot_w8a8"
    before_rot_vis_path = "/home/ecnu01/workspace/sure_quant/logs/a_vis_before_rot"

    os.makedirs(before_rot_vis_path, exist_ok=True)
    plt_llava_vision_activation(activations, output_path=before_rot_vis_path)

    print("==========================================")


def analyze_llava_activation_after_rot(vlm_llava):
    model = vlm_llava.model
    processor = vlm_llava.processor
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
    # raw_image = Image.open("/home/ccwan/stu_Jiangtp/MQuant/assert/sample1.jpg")
    # raw_image = Image.open("/home/ecnu01/workspace/awq_learn/sample_img/sample1.jpg")
    raw_image = Image.open("/home/ecnu01/workspace/awq_learn/sample_img/sample2.jpg")

    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)


    # paint activations vision
    activations = collect_llava_vision_input_activations(model, inputs)
    print(f"activations len: {len(activations)}")
    print(f"activations[0].shape: {activations[0].shape}")

    after_rot_vis_path = "/home/ecnu01/workspace/sure_quant/logs/a_vis_after_w4a4"
    # after_rot_vis_path = "/home/ecnu01/workspace/sure_quant/logs/a_vis_after_w8a8"
    os.makedirs(after_rot_vis_path, exist_ok=True)
    plt_llava_vision_activation(activations, output_path=after_rot_vis_path)

    print("==========================================")


def analyze_llava_weight():
    checkpoint = "/home/ecnu01/workspace/models/llava-1.5-7b-hf"
    # original model
    model = LlavaForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float16).eval()

    before_rot_vis_weight_path = "/home/ecnu01/workspace/sure_quant/logs/w_vis_before_rot"
    os.makedirs(before_rot_vis_weight_path, exist_ok=True)
    plt_llava_vision_weight(model, output_path=before_rot_vis_weight_path)

    print(">>>>>>>> done")




def analyze_llava_weight_after_rot(vlm_llava):
    # after_rot_vis_weight_path = "/home/ecnu01/workspace/sure_quant/logs/w_vis_after_rot_w8a8"
    after_rot_vis_weight_path = "/home/ecnu01/workspace/sure_quant/logs/w_vis_after_rot_w4a4"

    os.makedirs(after_rot_vis_weight_path, exist_ok=True)
    plt_llava_vision_weight(vlm_llava.model, output_path=after_rot_vis_weight_path)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llava-7B-Instruct")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--quant", action="store_true")

    # Rotation Arguments
    parser.add_argument(
        "--rotate", action="store_true", default=False, help="""Rotate the moodel. """
    )
    parser.add_argument(
        "--analysis", action="store_true", default=False, help="""analysis act. """
    )
    parser.add_argument(
        "--analysis_c_proj",
        action="store_true",
        default=False,
        help="""analysis act. """,
    )
    parser.add_argument(
        "--draw_save_path",
        type=str,
        default="output/llava_base",
        help="""analysis act save path. """,
    )
    parser.add_argument(
        "--rotate_visual_clip",
        action="store_true",
        default=False,
        help="""Rotate the moodel. """,
    )
    parser.add_argument(
        "--rotate_visual_cross_attn",
        action="store_true",
        default=False,
        help="""Rotate the moodel. """,
    )
    parser.add_argument(
        "--rotate_llm",
        action="store_true",
        default=False,
        help="""Rotate the moodel. """,
    )
    parser.add_argument(
        "--rotate_mode", type=str, default="hadamard", choices=["hadamard", "random"]
    )

    # Activation Quantization Arguments
    parser.add_argument(
        "--visual_a_bits",
        type=int,
        default=8,
        help="""Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)""",
    )
    # Activation Quantization Arguments
    parser.add_argument(
        "--llm_a_bits",
        type=int,
        default=8,
        help="""Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)""",
    )
    parser.add_argument(
        "--a_groupsize",
        type=int,
        default=-1,
        help="Groupsize for activation quantization. Note that this should be the same as w_groupsize",
    )
    parser.add_argument(
        "--a_asym",
        action="store_true",
        default=False,
        help="ASymmetric Activation quantization (default: False)",
    )
    parser.add_argument(
        "--a_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for activation quantization. new_max = max * clip_ratio",
    )

    # Weight Quantization Arguments
    parser.add_argument(
        "--visual_w_bits",
        type=int,
        default=4,
        help="Number of bits for weights of the Linear layers",
    )
    parser.add_argument(
        "--llm_w_bits",
        type=int,
        default=4,
        help="Number of bits for weights of the Linear layers",
    )
    parser.add_argument(
        "--w_groupsize",
        type=int,
        default=-1,
        help="Groupsize for weight quantization. Note that this should be the same as a_groupsize",
    )
    parser.add_argument(
        "--w_asym",
        action="store_true",
        default=False,
        help="ASymmetric weight quantization (default: False)",
    )
    parser.add_argument(
        "--visual_w_rtn",
        action="store_true",
        default=False,
        help="Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ",
    )
    parser.add_argument(
        "--llm_w_rtn",
        action="store_true",
        default=False,
        help="Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ",
    )
    parser.add_argument(
        "--visual_w_clip",
        action="store_true",
        default=False,
        help="""Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization""",
    )
    parser.add_argument(
        "--llm_w_clip",
        action="store_true",
        default=False,
        help="""Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization""",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--act_order", action="store_true", default=False, help="act-order in GPTQ"
    )
    parser.add_argument("--seed", type=int, default=42, help="seed")

    # General Quantization Arguments
    parser.add_argument(
        "--int8_down_proj",
        action="store_true",
        default=False,
        help="Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8",
    )

    parser.add_argument(
        "--quant_llm",
        action="store_true",
        default=False,
        help="Quantize the InternVL2-8B llm model",
    )

    parser.add_argument(
        "--quant_visual_clip",
        action="store_true",
        default=False,
        help="Quantize the visual features model",
    )

    parser.add_argument(
        "--quant_cross_attention",
        action="store_true",
        default=False,
        help="Quantize the cross attention model",
    )

    parser.add_argument(
        "--act_per_tensor",
        action="store_true",
        default=False,
        help="Quantize the activations per tensor",
    )

    parser.add_argument(
        "--nsamples",
        type=int,
        default=8,
        help="Number of calibration data samples for GPTQ.",
    )

    parser.add_argument(
        "--skip_names",
        nargs="+",
        default=[],
        help="Skip the quantization of the layers with these names",
    )

    parser.add_argument(
        "--no_fuse_visual_clip",
        action="store_true",
        default=False,
        help="Quantize the InternVL2-8B llm model",
    )

    parser.add_argument(
        "--no_fuse_visual_cross_attn",
        action="store_true",
        default=False,
        help="Quantize the visual features model",
    )

    parser.add_argument(
        "--no_fuse_llm",
        action="store_true",
        default=False,
        help="Quantize the cross attention model",
    )
    parser.add_argument(
        "--not_fuse_layer_norms",
        action="store_true",
        default=False,
        help="Quantize the cross attention model",
    )
    parser.add_argument(
        "--llm_static",
        action="store_true",
        default=False,
        help="quant act with static scale and zero point",
    )

    parser.add_argument(
        "--visual_static",
        action="store_true",
        default=False,
        help="quant act with static scale and zero point",
    )

    parser.add_argument(
        "--calib_num",
        type=int,
        default=32,
        help="calibration number",
    )

    parser.add_argument(
        "--eval_num",
        type=int,
        default=32,
        help="evaluation number",
    )

    parser.add_argument(
        "--calib_mode",
        type=str,
        default="v2",
        help="calibration mode, v1 or v2",
    )

    parser.add_argument(
        "--analysis_num",
        type=int,
        default=32,
        help="analysis number",
    )

    parser.add_argument(
        "--analysis_mode",
        type=str,
        default="v1",
        help="analysis mode, v1 or v2",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="TextVQA_VAL",
        help="dataset name",
    )
    parser.add_argument(
        "--analysis_text",
        action="store_true",
        default=False,
        help="analysis text",
    )
    parser.add_argument(
        "--online_visual_hadamard",
        action="store_true",
        default=False,
        help="Online Hadamard rotation",
    )

    parser.add_argument(
        "--online_llm_hadamard",
        action="store_true",
        default=False,
        help="Online Hadamard rotation",
    )
    parser.add_argument(
        "--fp32_had",
        action="store_true",
        default=False,
        help="Apply Hadamard rotation in FP32 (default: False)",
    )
    parser.add_argument(
        "--dump_gptq",
        type=str,
        default=None,
        help="Dump the GPTQ model to this path",
    )
    parser.add_argument(
        "--load_gptq",
        type=str,
        default=None,
        help="Load the GPTQ model from this path",
    )
    parser.add_argument(
        "--visual_split",
        action="store_true",
        default=False,
        help="visual split",
    )
    parser.add_argument(
        "--llm_split",
        action="store_true",
        default=False,
        help="Online Hadamard rotation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="max_new_tokens",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="verbose question and output",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        default=False,
        help="use real quantization"
    )
    parser.add_argument(
        "--real_mllm",
        action="store_true",
        default=False,
        help="use real quantization"
    )
    parser.add_argument(
        "--test_static",
        action="store_true",
        default=False,
        help="use real quantization"
    )
    parser.add_argument(
        "--test_time",
        action="store_true",
        default=False,
        help="use real quantization"
    )
    parser.add_argument(
        "--aifs",
        action="store_true",
        default=False,
        help="use aifs"
    )
    parser.add_argument(
        "--ttif",
        action="store_true",
        default=False,
        help="test ttif"
    )
    parser.add_argument(
        "--multi_moda",
        action="store_true",
        default=False,
        help="test multi_moda"
    )
    args = parser.parse_args()
    # print(args)

    # 未旋转量化，llava原始模型
    # analyze_llava_weight()
    # analyze_llava_activation()

    # 旋转量化模型，可视化分析权重、激活
    # main_w_analyze(args)
    # main_a_analyze(args)



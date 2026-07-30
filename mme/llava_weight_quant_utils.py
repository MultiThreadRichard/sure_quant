import math
import time
import tqdm
import torch
import torch.nn as nn
from fake_quant import utils
from fake_quant import quant_utils

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def llava_visual_clip_rtn(model, dev, args, quantizers):
    print("-----Rtn Quantization visual clip---")

    # visual clip
    layers = model.vision_tower.vision_model.encoder.layers
    torch.cuda.empty_cache()
    for i in tqdm.tqdm(
        range(len(layers)), desc="(RtN Quant.) visual clip"
    ):
        layer = layers[i]

        subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        for name in subset:
            if any(p_name in name for p_name in args.skip_names) or "L1" in name:
                continue
            layer_weight_bits = args.visual_w_bits
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits,
                perchannel=True,
                sym=not (args.w_asym),
                mse=args.visual_w_clip,
            )
            W = subset[name].weight.data
            dtype = W.dtype

            # print(W.data_ptr())
            # print(subset[name].weight.data.data_ptr())

            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(dtype)
            quantizers["model.vision_tower.vision_model.encoder.layers.%d.%s" % (i, name)] = quantizer.cpu()
        torch.cuda.empty_cache()



def llava_mm_projector_rtn(model, dev, args, quantizers):
    print("-----Rtn Quantization visual multi_modal_projector-----")
    subset = quant_utils.find_qlayers(model.multi_modal_projector, layers=[torch.nn.Linear])
    for name in subset:
        layer_weight_bits = args.visual_w_bits
        quantizer = quant_utils.WeightQuantizer()
        quantizer.configure(
            layer_weight_bits,
            perchannel=True,
            sym=not (args.w_asym),
            mse=args.visual_w_clip,
        )
        W = subset[name].weight.data
        dtype = W.dtype
        quantizer.find_params(W)
        subset[name].weight.data = quantizer.quantize(W).to(dtype)
        quantizers["model.multi_modal_projector.%s" % name] = quantizer.cpu()



def llava_llm_rtn(model, dev, args, quantizers):
    print("-----Rtn Quantization llm---")
    layers = model.language_model.model.layers
    torch.cuda.empty_cache()

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) LLM Layers"):
        layer = layers[i]

        subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        for name in subset:
            # if any(p_name in name for p_name in args.skip_names) or "L1" in name:
            #     continue
            layer_weight_bits = args.llm_w_bits
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits,
                perchannel=True,
                sym=not (args.w_asym),
                mse=args.llm_w_clip,
            )
            W = subset[name].weight.data
            dtype = W.dtype
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(dtype)
            quantizers["model.language_model.model.layers.%d.%s" % (i, name)] = quantizer.cpu()
        torch.cuda.empty_cache()



@torch.no_grad()
def llava_weight_quant_fwrd_plus(model, dataset, dev, dataset_name, args):
    print("-----RTN Quantization-----")

    quantizers = dict()

    if args.quant_visual_clip:
        if args.visual_w_rtn:
            llava_visual_clip_rtn(model.model, dev, args, quantizers)

    if args.quant_cross_attention:
        if args.visual_w_rtn:
            llava_mm_projector_rtn(model.model, dev, args, quantizers)


    if args.quant_llm:
        if args.llm_w_rtn:
            llava_llm_rtn(model.model, dev, args, quantizers)

    return quantizers

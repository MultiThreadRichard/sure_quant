import fast_hadamard_transform

from tqdm import tqdm
import math
import torch, torch.nn as nn, torch.nn.functional as F
from fake_quant import quant_utils
from fake_quant import hadamard_utils

# 导入 FP4 量化器
from int4_fp4_quantizer import FP4SingleQuantizer


class FP4ActQuantWrapper(torch.nn.Module):
    """
    使用 FP4Quantizer 的激活量化包装器
    """

    def __init__(self, module: torch.nn.Linear, act_per_tensor=False):
        super(FP4ActQuantWrapper, self).__init__()
        assert isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d))
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        
        self.quantizer = FP4SingleQuantizer(group_size=16)
        
        self.register_buffer("had_K", torch.tensor(0))
        self._buffers["had_K"] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False
        self.split = False
        self.static = False

    def extra_repr(self) -> str:
        return f"FP4 Activation Quantizer (group_size={self.quantizer.group_size})"

    def split_weights(self):
        self.L1 = torch.nn.Linear(1, self.module.out_features, bias=False).to(
            self.module.weight.device
        )
        self.L2 = torch.nn.Linear(
            self.module.in_features - 1,
            self.module.out_features,
            bias=True if self.module.bias is not None else False,
        ).to(self.module.weight.device)
        self.L1.weight.data = self.module.weight.data[:, 0:1]
        self.L2.weight.data = self.module.weight.data[:, 1:]
        if self.module.bias is not None:
            self.L2.bias.data = self.module.bias.data

    def forward(self, x):
        x_dtype = x.dtype

        # Rotate, if needed
        if self.online_full_had:
            if self.fp32_had:
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(
                    x_dtype
                )
            else:
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)

        elif self.online_partial_had:
            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            if self.K == 1:
                x = fast_hadamard_transform.hadamard_transform(
                    x.reshape(
                        -1, init_shape[-1] // self.had_dim, self.had_dim
                    ).transpose(1, 2),
                    scale=1 / math.sqrt(init_shape[-1] // self.had_dim),
                ).transpose(1, 2)
            else:
                x = (
                    self.had_K.to(x.dtype)
                    @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)
                ) / math.sqrt(init_shape[-1] // self.had_dim)

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        if self.split:
            if self.static:
                x[..., 1:] = self.quantizer.dequantize(self.quantizer.quantize(x[..., 1:]))
            else:
                compressed = self.quantizer.quantize(x[..., 1:])
                x[..., 1:] = self.quantizer.dequantize(compressed).to(x_dtype)
            x1 = self.L1.float()(x[..., 0:1].float())
            x2 = self.L2.float()(x[..., 1:].float())
            x = (x1 + x2).to(x_dtype)
        else:
            # 使用 FP4 量化激活
            compressed = self.quantizer.quantize(x)
            x = self.quantizer.dequantize(compressed).to(x_dtype)
            
            x = self.module(x).to(x_dtype)

        return x


def add_actquant_fp4(module, act_per_tensor=False):
    """
    使用 FP4ActQuantWrapper 添加激活量化
    """
    
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, FP4ActQuantWrapper(child, act_per_tensor))
        else:
            add_actquant_fp4(child, act_per_tensor)


def llava_add_act_quant_fp4(model, args):
    """
    FP4 版本的 llava_add_act_quant - 使用 FP4Quantizer
    """
    if args.quant_llm:
        add_actquant_fp4(
            model.model.language_model.model,
            args.act_per_tensor,
        )

    if args.quant_visual_clip:
        add_actquant_fp4(model.model.vision_tower.vision_model.encoder.layers, args.act_per_tensor)

    if args.quant_cross_attention:
        add_actquant_fp4(model.model.multi_modal_projector, args.act_per_tensor)


# ============================================================
# FP4 权重量化函数
# ============================================================

def llava_visual_clip_rtn_fp4(model, dev, args, quantizers):
    """
    使用 FP4Quantizer 对视觉编码器进行权重量化
    """
    print("-----FP4 Rtn Quantization visual clip---")
    layers = model.vision_tower.vision_model.encoder.layers
    torch.cuda.empty_cache()
    
    for i in tqdm(
        range(len(layers)), desc="(FP4 RtN Quant.) visual clip"
    ):
        layer = layers[i]
        subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        for name in subset:
            if any(p_name in name for p_name in args.skip_names) or "L1" in name:
                continue
            
            W = subset[name].weight.data
            dtype = W.dtype
            
            quantizer = FP4SingleQuantizer(group_size=args.w_groupsize if args.w_groupsize > 0 else 16)

            compressed = quantizer.quantize(W)
            dequantized = quantizer.dequantize(compressed)
            
            subset[name].weight.data = dequantized.to(dtype)
            quantizers["model.vision_tower.vision_model.encoder.layers.%d.%s" % (i, name)] = quantizer
        
        torch.cuda.empty_cache()


def llava_mm_projector_rtn_fp4(model, dev, args, quantizers):
    """
    使用 FP4Quantizer 对多模态投影器进行权重量化
    """
    print("-----FP4 Rtn Quantization visual multi_modal_projector-----")
    subset = quant_utils.find_qlayers(model.multi_modal_projector, layers=[torch.nn.Linear])
    
    for name in subset:
        W = subset[name].weight.data
        dtype = W.dtype
        
        # 使用 FP4Quantizer 量化权重
        # quantizer = FP4Quantizer(group_size=args.w_groupsize if args.w_groupsize > 0 else 16)
        quantizer = FP4SingleQuantizer(group_size=args.w_groupsize if args.w_groupsize > 0 else 16)

        compressed = quantizer.quantize(W)
        dequantized = quantizer.dequantize(compressed)
        
        subset[name].weight.data = dequantized.to(dtype)
        quantizers["model.multi_modal_projector.%s" % name] = quantizer


def llava_llm_rtn_fp4(model, dev, args, quantizers):
    """
    使用 FP4Quantizer 对 LLM 进行权重量化
    """
    print("-----FP4 Rtn Quantization llm---")
    layers = model.language_model.model.layers
    torch.cuda.empty_cache()

    for i in tqdm(range(len(layers)), desc="(FP4 RtN Quant.) LLM Layers"):
        layer = layers[i]
        subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        for name in subset:
            W = subset[name].weight.data
            dtype = W.dtype
            
            # 使用 FP4Quantizer 量化权重
            # quantizer = FP4Quantizer(group_size=args.w_groupsize if args.w_groupsize > 0 else 16)
            quantizer = FP4SingleQuantizer(group_size=args.w_groupsize if args.w_groupsize > 0 else 16)

            compressed = quantizer.quantize(W)
            dequantized = quantizer.dequantize(compressed)
            
            subset[name].weight.data = dequantized.to(dtype)
            quantizers["model.language_model.model.layers.%d.%s" % (i, name)] = quantizer
        
        torch.cuda.empty_cache()


@torch.no_grad()
def llava_weight_quant_fwrd_plus_fp4(model, dataset, dev, dataset_name, args):
    """
    FP4 版本的 llava_weight_quant_fwrd_plus - 使用 FP4Quantizer
    """
    print("-----FP4 RTN Quantization-----")

    quantizers = dict()

    if args.quant_visual_clip:
        llava_visual_clip_rtn_fp4(model.model, dev, args, quantizers)

    if args.quant_cross_attention:
        llava_mm_projector_rtn_fp4(model.model, dev, args, quantizers)

    if args.quant_llm:
        llava_llm_rtn_fp4(model.model, dev, args, quantizers)
    
    return quantizers
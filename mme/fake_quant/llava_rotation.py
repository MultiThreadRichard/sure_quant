import torch
import tqdm
import typing
from fake_quant.rotation_utils import (
    fuse_ln_linear,
    bake_mean_into_conv,
    bake_mean_into_linear,
    rotate_conv,
)
from fake_quant.rotation_utils import get_orthogonal_matrix
from fake_quant import module_util
from fake_quant import utils
from fake_quant.hadamard_utils import apply_exact_had_to_linear


def fuse_merger_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        w_o, w_i = W_.shape
        size = layernorm.weight.shape[0]
        linear.weight.data = (
            (W_.view(w_o, -1, size) * layernorm.weight.double())
            .to(linear_dtype)
            .view(w_o, w_i)
        )

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64).to(W_)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_.view(w_o, -1, size), layernorm.bias.double()
            ).sum(dim=-1)
            linear.bias.data = linear.bias.data.to(linear_dtype)

    layernorm.weight.data = torch.ones_like(layernorm.weight.data)
    if hasattr(layernorm, "bias"):
        layernorm.bias.data = torch.zeros_like(layernorm.bias.data)


def fuse_ln_ln(ln_a: torch.nn.LayerNorm, ln_b: torch.nn.LayerNorm):
    """
    融合 连续两个 LayerNorm：pre_layrnorm → layer_norm1
    保证输出完全不变
    """
    with torch.no_grad():
        # 数学等价合并：LN_a → LN_b 合并为新 LN_b
        gamma = ln_a.weight * ln_b.weight
        beta = ln_a.bias * ln_b.weight + ln_b.bias
        ln_b.weight.data.copy_(gamma)
        ln_b.bias.data.copy_(beta)



# model: tranformers model 包装了一个类，model.model是tranformers库模型
def fuse_llava_layer_norms(model, args):
    print("fuse llava layer norms")
    if not args.no_fuse_visual_clip:
        # bake_mean_into_conv(model.model.vision_tower.vision_model.embeddings.patch_embedding)

        # 步骤1：pre_layrnorm 融合进 第0层 layer_norm1
        # fuse_ln_ln(model.model.vision_tower.vision_model.pre_layrnorm, model.model.vision_tower.vision_model.encoder.layers[0].layer_norm1)

        # 融合后必须关掉 pre_layrnorm
        # model.model.vision_tower.vision_model.pre_layrnorm = torch.nn.Identity()
        # model.model.vision_tower.vision_model.pre_layrnorm = module_util.RMSN(
        #     model.model.vision_tower.vision_model.embeddings.embed_dim, eps=1e-6
        # )

        # print(model.model.vision_tower.vision_model.pre_layrnorm)
        # print(model.model.vision_tower.vision_model.encoder.layers[0].layer_norm1)

        for layer in model.model.vision_tower.vision_model.encoder.layers:
            fuse_ln_linear(
                layer.layer_norm1, 
                [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ]
            )
            
            fuse_ln_linear(layer.layer_norm2, [layer.mlp.fc1])

            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.mlp.fc2)
        
        # # 2. 【关键修复】pre_layrnorm 融合进第一层 q/k/v
        # first_layer = model.model.vision_tower.vision_model.encoder.layers[0]
        # fuse_ln_linear(
        #     model.model.vision_tower.vision_model.pre_layrnorm,
        #     [
        #         first_layer.self_attn.q_proj,
        #         first_layer.self_attn.k_proj,
        #         first_layer.self_attn.v_proj,
        #     ]
        # )
        # bake_mean_into_linear(first_layer.self_attn.out_proj)


        # # 融合后必须关掉 pre_layrnorm
        # # model.model.vision_tower.vision_model.pre_layrnorm = torch.nn.Identity()
        # model.model.vision_tower.vision_model.pre_layrnorm = module_util.RMSN(
        #     model.model.vision_tower.vision_model.embeddings.embed_dim, eps=1e-6
        # )


        module_util.replace_modules(
            model.model.vision_tower.vision_model.encoder.layers,
            torch.nn.LayerNorm,
            lambda _: module_util.RMSN(
                model.model.vision_tower.vision_model.embeddings.embed_dim, eps=1e-6
            ),
            replace_layers=False,
        )

    # # TODO CHECK
    if not args.no_fuse_visual_cross_attn:
        fuse_merger_linear(
            model.model.vision_tower.vision_model.post_layernorm, [model.model.multi_modal_projector.linear_1]
        )
        model.model.vision_tower.vision_model.post_layernorm = module_util.RMSN(
            model.model.vision_tower.vision_model.embeddings.embed_dim,
            eps=1e-6,
        )

        # bake_mean_into_linear(model.model.multi_modal_projector.linear_2)

        # module_util.replace_modules(
        #     model.model.vision_tower.vision_model.post_layernorm,
        #     torch.nn.LayerNorm,
        #     lambda _: module_util.RMSN(
        #         model.model.vision_tower.vision_model.embeddings.embed_dim,
        #         eps=1e-6,
        #     ),
        #     replace_layers=False,
        # )

    if not args.no_fuse_llm:
        for layer in model.model.language_model.model.layers:
            fuse_ln_linear(
                layer.input_layernorm,
                [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ],
            )
            fuse_ln_linear(
                layer.post_attention_layernorm, [layer.mlp.gate_proj, layer.mlp.up_proj]
            )

        # 最后一个rmsnorm需要和lm_head合并
        fuse_ln_linear(model.model.language_model.model.norm, [model.model.language_model.lm_head])

# only language
# def fuse_llava_layer_norms(model, args):
#     from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaSdpaAttention
#     print("fuse llava layer norms")
#     if not args.no_fuse_llm:
        
#         # Embedding fusion
#         # print(model.model.language_model.model.embed_tokens.weight.data)
#         # for W in [model.model.language_model.model.embed_tokens]:
#         #     W_ = W.weight.data.double()
#         #     W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
#         # print(model.model.language_model.model.embed_tokens.weight.data)
        

#         for layer in model.model.language_model.model.layers:
#             fuse_ln_linear(
#                 layer.input_layernorm,
#                 [
#                     layer.self_attn.q_proj,
#                     layer.self_attn.k_proj,
#                     layer.self_attn.v_proj,
#                 ],
#             )
#             fuse_ln_linear(
#                 layer.post_attention_layernorm, [layer.mlp.gate_proj, layer.mlp.up_proj]
#             )

#         # 最后一个rmsnorm需要和lm_head合并
#         fuse_ln_linear(model.model.language_model.model.norm, [model.model.language_model.lm_head])

#         # module_util.replace_modules(
#         #     model.model.language_model.model,
#         #     LlamaRMSNorm,
#         #     lambda _: module_util.RMSN(
#         #         model.model.language_model.config.hidden_size, eps=1e-6
#         #     ),
#         #     replace_layers=False,
#         # )


def rotate_llava_attention_inputs(layer, Q, is_visual=False) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.

    layer_list = (
        [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
        # if not is_visual
        # else [layer.attn.qkv]
    )
    for W in layer_list:
        dtype = W.weight.dtype
        W_ = W.weight.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)
        # if W.bias is not None:
        #     b = W.bias.data.to(dtype=torch.float64)
        #     W.bias.data = torch.matmul(b, Q).to(dtype=dtype)

def rotate_llava_attention_output(layer, Q, is_visual=False) -> None:
    # Rotate output matrix of the self-attention layer.
    if is_visual:
        W = layer.self_attn.out_proj
    else:
        W = layer.self_attn.o_proj

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)


def rotate_llava_mlp_input(layer, Q, is_visual=False) -> None:
    # Rotate the MLP input weights.
    if is_visual:
        mlp_inputs = [layer.mlp.fc1]
    else:
        mlp_inputs = [layer.mlp.gate_proj, layer.mlp.up_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)
        # if W.bias is not None:
        #     b = W.bias.data.to(dtype=torch.float64)
        #     W.bias.data = torch.matmul(b, Q).to(dtype=dtype)


def rotate_llava_mlp_output(layer, Q, is_visual=False, online_hadamard=False):
    out_layer = layer.mlp.fc2 if is_visual else layer.mlp.down_proj
    # Rotate the MLP output weights and bias.
    dtype = out_layer.weight.data.dtype
    W_ = out_layer.weight.data.to(dtype=torch.float64)
    out_layer.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)

    if online_hadamard:
        # apply exact (inverse) hadamard on the weights of mlp output
        apply_exact_had_to_linear(
            out_layer, had_dim=-1, output=False
        )

    if out_layer.bias is not None:
        b = out_layer.bias.data.to(dtype=torch.float64)
        out_layer.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)


def rotate_llava_ov_proj(layer, head_num, head_dim, is_visual=False):
    if is_visual:
        # pass
        # LLava 结构：self_attn 分开有 q_proj, k_proj, v_proj, out_proj
        # no online
        
        # dtype = vproj.weight.data.dtype
        # W_ = vproj.weight.data.to(dtype=torch.float64)
        # vproj.weight.data = torch.matmul(W_, Q).to(dtype=dtype)

        ###########
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.out_proj

        # 获取正交矩阵（hadamard）
        Q = get_orthogonal_matrix(head_dim, mode="hadamard")
        dtype = v_proj.weight.dtype

        # ==================== 旋转 v_proj ====================
        # 处理 v_proj 权重
        W_v = v_proj.weight.data.to(dtype=torch.float64)
        # 形状变换: (out_features, in_features) → (in_features, out_features)
        W_v = W_v.T.reshape(-1, head_num, head_dim)
        # 矩阵旋转
        v_proj.weight.data = torch.matmul(W_v, Q).reshape(-1, head_num * head_dim).T.to(dtype=dtype)

        # 处理 v_proj 偏置
        if v_proj.bias is not None:
            b_v = v_proj.bias.data.to(dtype=torch.float64).reshape(head_num, head_dim)
            v_proj.bias.data = torch.matmul(b_v, Q).to(dtype=dtype).reshape(-1)

        # ==================== 旋转 out_proj (o_proj) ====================
        W_o = o_proj.weight.data.to(dtype=torch.float64).reshape(-1, head_num, head_dim)
        o_proj.weight.data = torch.matmul(W_o, Q).reshape(-1, head_num * head_dim).to(dtype=dtype)

    else:
        # online
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj
        apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
        apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False)

        # apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
        # apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)

        # apply_exact_had_to_linear(v_proj, had_dim=-1, output=True)
        # apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False)

        # apply_exact_had_to_linear(v_proj, had_dim=-1, output=True)
        # apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)




def rotate_visual_merger(model, Q: torch.Tensor) -> None:
    Q = Q.to(model.multi_modal_projector.linear_1.weight.device)

    # Rotate the head.
    dtype = model.multi_modal_projector.linear_1.weight.dtype

    q_shape = Q.shape[0]
    o_shape, i_shape = model.multi_modal_projector.linear_1.weight.shape

    W_ = (
        model.multi_modal_projector.linear_1
        .weight.to(dtype=torch.float64)
        .reshape(o_shape, -1, q_shape)
    )
    model.multi_modal_projector.linear_1.weight.data = (
        torch.matmul(W_, Q).to(dtype=dtype).reshape(o_shape, i_shape).contiguous()
    )


def rotate_llava_embeddings_mmproj(model, Q) -> None:
    Q = Q.to(model.language_model.model.embed_tokens.weight.device)
    dtype = model.language_model.model.embed_tokens.weight.data.dtype
    W_ = model.language_model.model.embed_tokens.weight.data.to(dtype=torch.float64)
    model.language_model.model.embed_tokens.weight.data = torch.matmul(W_, Q).to(dtype=dtype)

    Q = Q.to(model.multi_modal_projector.linear_2.weight.device)
    W_ = model.multi_modal_projector.linear_2.weight.data.to(dtype=torch.float64)
    model.multi_modal_projector.linear_2.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
    # TODO CHECK
    if model.multi_modal_projector.linear_2.bias is not None:
        b = model.multi_modal_projector.linear_2.bias.data.to(dtype=torch.float64)
        # print(b.shape, Q.shape)
        # print(torch.matmul(b, Q).shape)
        # print(torch.matmul(Q.T, b).shape)

        # model.multi_modal_projector.linear_2.bias.data = torch.matmul(b, Q).to(dtype=dtype)
        model.multi_modal_projector.linear_2.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)



def rotate_llava_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    dtype = model.language_model.lm_head.weight.data.dtype
    W_ = model.language_model.lm_head.weight.data.to(dtype=torch.float64)
    model.language_model.lm_head.weight.data = torch.matmul(W_, Q.to(W_.device)).to(dtype=dtype)


# def rotate_vision_pre_layernorm(pre_layernorm, Q):
#     Q = Q.to(pre_layernorm.weight.device)

#     dtype = pre_layernorm.weight.data.dtype
#     print(f"pre_layernorm.weight.shape: {pre_layernorm.weight.shape}")
#     # W_ = pre_layernorm.weight.data.to(dtype=torch.float64)
#     # pre_layernorm.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
#     # # pre_layernorm.weight.data = torch.matmul(W_, Q).to(dtype=dtype)

#     # if pre_layernorm.bias is not None:
#     #     b = pre_layernorm.bias.data.to(dtype=torch.float64)
#     #     print(f"b.shape: {b.shape}")
#     #     # print(torch.matmul(b, Q).shape)
#     #     # print(torch.matmul(Q.T, b).shape)

#     #     # pre_layernorm.bias.data = torch.matmul(b, Q).to(dtype=dtype)
#     #     pre_layernorm.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)
    
#     # ================= 正确操作：权重 右乘 Q =================
#     # W = pre_layernorm.weight.data.to(dtype=torch.float64).unsqueeze(0)
#     # print(f"W.shape: {W.shape}, Q.shape: {Q.shape}")
#     # # W @ Q → [D, D]
#     # W_new = torch.matmul(W, Q).squeeze(0)
#     # print(f"W_new.shape: {W_new.shape}")

#     # pre_layernorm.weight.data = W_new.to(dtype=dtype)

#     # # bias 同样 右乘 Q
#     # if pre_layernorm.bias is not None:
#     #     b = pre_layernorm.bias.data.to(dtype=torch.float64).unsqueeze(0)
#     #     print(f"b.shape: {b.shape}")
#     #     b_new = torch.matmul(b, Q).squeeze(0)
#     #     pre_layernorm.bias.data = b_new.to(dtype=dtype)

#     W = pre_layernorm.weight.data.to(dtype=torch.float64)
#     print(f"W.shape: {W.shape}, Q.shape: {Q.shape}")
#     # W @ Q → [D, D]
#     W_new = torch.matmul(W, Q)
#     print(f"W_new.shape: {W_new.shape}")

#     pre_layernorm.weight.data = W_new.to(dtype=dtype)

#     # bias 同样 右乘 Q
#     if pre_layernorm.bias is not None:
#         b = pre_layernorm.bias.data.to(dtype=torch.float64)
#         print(f"b.shape: {b.shape}")
#         b_new = torch.matmul(b, Q)
#         pre_layernorm.bias.data = b_new.to(dtype=dtype)


def rotate_vision_pre_layernorm(pre_layernorm, Q, dev):
    Q = Q.to(dev)

    def hook_fn(module, input, output):
        # bake mean
        dtype = output.data.dtype
        Out = output.data.double()
        Out = Out - Out.mean(dim=-1, keepdim=True)
        output.data = Out.to(dtype=dtype)

        # rotate
        # print(f">>>> hook  output.shape: {output.shape}, Q.shape: {Q.shape}")
        # print(f"type(output), type(Q): {type(output)}, {type(Q)}") #  <class 'torch.Tensor'>, <class 'torch.Tensor'>
        Out = output.data.to(dtype=torch.float64)
        output.data = torch.matmul(Out, Q).to(dtype=dtype)

        # Out = output.data.double()
        # Out = Out - Out.mean(dim=-1, keepdim=True)
        # output.data = Out.to(dtype=dtype)
        return output

    # 注册 hook
    handle = pre_layernorm.register_forward_hook(hook_fn)
    return handle  # 可以用 handle.remove() 删掉 hook



@torch.no_grad()
def rotate_llava_model(model, args):
    '''
    model: tranformers库模型
    '''
    print("rotate model")
    Q_v = None

    if args.rotate_visual_clip:
        # rotate visual transformer
        vision_config = model.vision_tower.config
        num_heads = vision_config.num_attention_heads
        head_dim = vision_config.hidden_size // num_heads
        Q_v = get_orthogonal_matrix(
            vision_config.hidden_size, args.rotate_mode
        )

        print(f"vision_config.hidden_size: {vision_config.hidden_size}")

        # TODO CHECK
        # rotate_conv(
        #     model.vision_tower.vision_model.embeddings.patch_embedding,
        #     Q_v,
        #     vision_config.hidden_size,
        # )
        # rotate_vision_pre_layernorm(model.vision_tower.vision_model.pre_layrnorm, Q_v)


        for idx, layer in enumerate(
            tqdm.tqdm(
                model.vision_tower.vision_model.encoder.layers,
                unit="layer",
                desc="Rotating Visual CLIP",
            )
        ):
            layer_device = next(layer.parameters()).device
            Q_v = Q_v.to(layer_device)
            rotate_llava_attention_inputs(layer, Q_v, is_visual=True)
            rotate_llava_attention_output(layer, Q_v, is_visual=True)
            rotate_llava_mlp_input(layer, Q_v, is_visual=True)
            rotate_llava_mlp_output(
                layer,
                Q_v,
                True,
                args.online_visual_hadamard,
            )

            rotate_llava_ov_proj(
                layer,
                num_heads,
                head_dim,
                is_visual=True,
            )

        rotate_visual_merger(model, Q_v)
        utils.cleanup_memory()

    # if args.rotate_visual_cross_attn:
    #     print("\n Rotating Visual Cross Attention \n")
    #     pass

    if args.rotate_llm:
        if args.online_llm_hadamard:
            model.config.need_pad = False

            lang_config = model.language_model.config

            # TODO CHECK
            from fake_quant.hadamard_utils import auto_pad_size

            new_intermediate_size = auto_pad_size(lang_config.intermediate_size)
            # print(f"model.config.need_pad: {model.config.need_pad}")
            print(f"new_intermediate_size: {new_intermediate_size}")

            if new_intermediate_size != lang_config.intermediate_size:
                for name, module in model.named_modules():
                    if "down_proj" in name and isinstance(module, torch.nn.Linear):
                        new_module = torch.nn.Linear(
                            new_intermediate_size,
                            module.out_features,
                            dtype=module.weight.dtype,
                        ).to(module.weight.device)
                        with torch.no_grad():
                            new_module.weight[:, : module.in_features] = (
                                module.weight.data
                            )
                            if module.bias is not None:
                                new_module.bias[: module.out_features].copy_(
                                    module.bias
                                )
                        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                        if parent_name:  # 如果模块不是顶层模块
                            parent = dict(model.named_modules())[parent_name]
                            setattr(parent, name.split(".")[-1], new_module)
                        else:  # 如果模块是顶层模块
                            setattr(model, name, new_module)
                model.language_model.config.intermediate_size = new_intermediate_size
                model.config.need_pad = True
        
        Q = get_orthogonal_matrix(lang_config.hidden_size, args.rotate_mode)

        num_attention_heads = lang_config.num_attention_heads
        num_key_value_head = lang_config.num_key_value_heads
        model_dim = lang_config.hidden_size
        head_dim = model_dim // num_attention_heads

        
        rotate_llava_embeddings_mmproj(model, Q)
        rotate_llava_head(model, Q)
        utils.cleanup_memory()
        for idx, layer in enumerate(
            tqdm.tqdm(model.language_model.model.layers, unit="layer", desc="Rotating LLM")
        ):
            # if idx != 0:
            #     continue
            layer_device = next(layer.parameters()).device
            Q = Q.to(layer_device)
            rotate_llava_attention_inputs(layer, Q)
            rotate_llava_attention_output(layer, Q)
            rotate_llava_mlp_input(layer, Q)
            rotate_llava_mlp_output(layer, Q, False, args.online_llm_hadamard)

            rotate_llava_ov_proj(
                layer, num_attention_heads, head_dim, is_visual=False
            )
        utils.cleanup_memory()
    return Q_v

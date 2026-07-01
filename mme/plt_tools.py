import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
# import numpy as np
# from PIL import Image



# -------------------------- 逐维度统计分位数 --------------------------
def calc_dimension_stats(activation):
    """输入形状 (token数, 维度数),axis=0 按矩阵的列压缩,返回每个维度的统计量"""
    min_val = np.min(activation, axis=0)
    max_val = np.max(activation, axis=0)
    p1 = np.percentile(activation, 1, axis=0)
    p99 = np.percentile(activation, 99, axis=0)
    p25 = np.percentile(activation, 25, axis=0)
    p75 = np.percentile(activation, 75, axis=0)
    return min_val, max_val, p1, p99, p25, p75


def plt_range_val_dim(np_tensor, output_path, title="Before Rot", xlabel="Hidden dimension index", ylabel="Weight value"):
    min_val, max_val, p1, p99, p25, p75 = calc_dimension_stats(np_tensor)
    hidden_size = np_tensor.shape[-1]
    x = np.arange(hidden_size)  # 每个隐藏维度对应一个x坐标

    # -------------------------- 3. 绘制分段竖线图 --------------------------
    plt.figure(figsize=(10, 8), dpi=120)
    ax = plt.gca()

    # 配色严格匹配原图
    color_25_75 = "#f2b138"   # 黄色：四分位区间
    color_1_99 = "#d93a6e"    # 玫红色：1/99分位区间
    color_minmax = "#367bc1"  # 蓝色：极值区间
    line_width = 0.6  # 细线宽，保证4096条线紧密排列成连续色带

    # 1. 中间黄色段：25% ~ 75% 分位
    ax.vlines(x, ymin=p25, ymax=p75, color=color_25_75, linewidth=line_width)
    # 2. 红色段：1%~25% 和 75%~99%
    ax.vlines(x, ymin=p1, ymax=p25, color=color_1_99, linewidth=line_width)
    ax.vlines(x, ymin=p75, ymax=p99, color=color_1_99, linewidth=line_width)
    # 3. 蓝色段：min~1% 和 99%~max
    ax.vlines(x, ymin=min_val, ymax=p1, color=color_minmax, linewidth=line_width)
    ax.vlines(x, ymin=p99, ymax=max_val, color=color_minmax, linewidth=line_width)

    # -------------------------- 4. 坐标轴与样式设置 --------------------------
    ax.set_title(title, fontsize=20, pad=15)
    ax.set_xlabel(xlabel, fontsize=18, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=18, labelpad=10)

    ax.set_ylim(-0.48, 0.48)
    ax.set_xlim(0, hidden_size)
    ax.tick_params(axis='both', labelsize=16)

    # 匹配原图样式的图例
    legend_elements = [
        Line2D([0], [0], color=color_minmax, lw=2, label='Min/Max'),
        Line2D([0], [0], color=color_1_99, lw=2, label='1/99 Percentile'),
        Line2D([0], [0], color=color_25_75, lw=2, label='25/75 Percentile')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)

    plt.tight_layout()
    # plt.show()

    
    plt.savefig(output_path, dpi=300)
    print(f"save to {output_path}")


def plt_act_val_dim(np_tensor, output_path, title="Before Rot", xlabel="Hidden dimension index", ylabel="Weight value"):
    min_val, max_val, p1, p99, p25, p75 = calc_dimension_stats(np_tensor)
    hidden_size = np_tensor.shape[-1]
    x = np.arange(hidden_size)  # 每个隐藏维度对应一个x坐标

    # -------------------------- 3. 绘制分段竖线图 --------------------------
    plt.figure(figsize=(10, 8), dpi=120)
    ax = plt.gca()

    # 配色严格匹配原图
    color_25_75 = "#f2b138"   # 黄色：四分位区间
    color_1_99 = "#d93a6e"    # 玫红色：1/99分位区间
    color_minmax = "#367bc1"  # 蓝色：极值区间
    line_width = 0.6  # 细线宽，保证4096条线紧密排列成连续色带

    # 1. 中间黄色段：25% ~ 75% 分位
    ax.vlines(x, ymin=p25, ymax=p75, color=color_25_75, linewidth=line_width)
    # 2. 红色段：1%~25% 和 75%~99%
    ax.vlines(x, ymin=p1, ymax=p25, color=color_1_99, linewidth=line_width)
    ax.vlines(x, ymin=p75, ymax=p99, color=color_1_99, linewidth=line_width)
    # 3. 蓝色段：min~1% 和 99%~max
    ax.vlines(x, ymin=min_val, ymax=p1, color=color_minmax, linewidth=line_width)
    ax.vlines(x, ymin=p99, ymax=max_val, color=color_minmax, linewidth=line_width)

    # -------------------------- 4. 坐标轴与样式设置 --------------------------
    ax.set_title(title, fontsize=20, pad=15)
    ax.set_xlabel(xlabel, fontsize=18, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=18, labelpad=10)

    # ax.set_ylim(-0.48, 0.48)
    # ax.set_xlim(0, hidden_size)
    ax.tick_params(axis='both', labelsize=16)

    # 匹配原图样式的图例
    legend_elements = [
        Line2D([0], [0], color=color_minmax, lw=2, label='Min/Max'),
        Line2D([0], [0], color=color_1_99, lw=2, label='1/99 Percentile'),
        Line2D([0], [0], color=color_25_75, lw=2, label='25/75 Percentile')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)

    plt.tight_layout()
    # plt.show()

    
    plt.savefig(output_path, dpi=300)
    print(f"save to {output_path}")


# # ===================== 可配置参数 =====================
# # 模型名称：可替换为 Llama-3、Mistral 等同类架构模型
# model_name = "meta-llama/Llama-2-7b-hf"
# layer_idx = 10  # 目标层号，与原图第10层对应
# # 权重模块路径：可选 q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj
# weight_path = "mlp.gate_proj.weight"
# # 统计维度：0=按输入维度统计（x轴为输入隐藏维度，和激活图对齐）；1=按输出维度统计
# stat_axis = 0
# # ======================================================


def plt_llava_lang_weight(model, output_path, stat_axis=0):
    attr_targets = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", 
                    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    
    # 从模型中提取权重张量
    for idx, layer in enumerate(model.language_model.model.layers):
        if idx > 10 and idx <= 29:
            continue

        for attr_name in attr_targets:
            target_module = layer

            for attr in attr_name.split("."):
                target_module = getattr(target_module, attr)
            
            # print(f"layer {idx} {attr_name}: {target_module.weight.shape}")
            weight_matrix = target_module.weight.T.detach().cpu().float().numpy()  # 转float32避免numpy计算精度问题
            
            print(f"layer {idx} {attr_name} weight_matrix: {weight_matrix.shape}")
            plt_range_val_dim(weight_matrix, f"{output_path}/layer_{idx}_{attr_name}_weight.png", title=f"Layer {idx} {attr_name} Weight")
        # break

    # target_module = model.model.layers[layer_idx]
    # for attr in weight_path.split("."):
    #     target_module = getattr(target_module, attr)
    # weight_matrix = target_module.detach().cpu().float().numpy()  # 转float32避免numpy计算精度问题
    # print(f"权重矩阵形状: {weight_matrix.shape}")
    # print(f"统计维度：axis={stat_axis}，共 {weight_matrix.shape[stat_axis]} 个维度")


def plt_llava_vision_weight(model, output_path, stat_axis=0):
    attr_targets = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", 
                    "mlp.fc1", "mlp.fc2"]
    
    # 从模型中提取权重张量
    for idx, layer in enumerate(model.vision_tower.vision_model.encoder.layers):
        if idx > 10 and idx < 22:
            continue
        # if idx != 22:
        #     continue

        for attr_name in attr_targets:
            target_module = layer

            for attr in attr_name.split("."):
                target_module = getattr(target_module, attr)
            
            # print(f"layer {idx} {attr_name}: {target_module.weight.shape}")
            if stat_axis == 0:
                weight_matrix = target_module.weight.T.detach().cpu().float().numpy()  # 转float32避免numpy计算精度问题
            else:
                weight_matrix = target_module.weight.detach().cpu().float().numpy()

            print(f"layer {idx} {attr_name} weight_matrix: {weight_matrix.shape}")
            plt_range_val_dim(weight_matrix, f"{output_path}/layer_{idx}_{attr_name}_weight.png", title=f"Layer {idx} {attr_name} Weight")
        # break




def collect_llava_lang_decoder_input_activations(
    model,
    inputs,
):
    """
    采集LLAVA解码器(language_model)每一层的输入激活, return ndarray
    """
    # 存储每层激活的容器
    layer_activations = []
    hook_handles = []

    # 定义钩子函数：捕获每层输入
    def layer_input_hook(layer_idx):
        def hook(module, input_args, output):
            # input_args[0] 是该层主输入张量，shape: [batch, seq_len, hidden_size]
            # print(f"input_args: {input_args}")
            input_act = input_args[0].detach().cpu().float()
            # 展平batch维度，统一为 [seq_len, hidden_dim]
            # print(f"layer {layer_idx} input_act.shape: {input_act.shape}")
            b, s, h = input_act.shape
            np_act = input_act.reshape(-1, h).numpy()
            layer_activations.append(np_act)
        return hook

    # 给解码器所有层注册钩子
    decoder_layers = model.language_model.model.layers
    num_layers = len(decoder_layers)
    for idx in range(num_layers):
        handle = decoder_layers[idx].register_forward_hook(layer_input_hook(idx))
        hook_handles.append(handle)

    # 推理
    with torch.no_grad():
        output = model(**inputs)

    # 移除所有钩子，防止后续污染
    for h in hook_handles:
        h.remove()

    return layer_activations


def collect_llava_vision_input_activations(
    model,
    inputs,
):
    # 存储每层激活的容器
    layer_activations = []
    hook_handles = []

    # 定义钩子函数：捕获每层输入
    def layer_input_hook(layer_idx):
        def hook(module, input_args, output):
            # input_args[0] 是该层主输入张量，shape: [batch, seq_len, hidden_size]
            # print(f"input_args: {input_args}")
            input_act = input_args[0].detach().cpu().float()
            # 展平batch维度，统一为 [seq_len, hidden_dim]
            # print(f"layer {layer_idx} input_act.shape: {input_act.shape}")
            b, s, h = input_act.shape
            np_act = input_act.reshape(-1, h).numpy()
            layer_activations.append(np_act)
        return hook

    # 注册钩子
    target_layers = model.vision_tower.vision_model.encoder.layers
    num_layers = len(target_layers)
    for idx in range(num_layers):
        handle = target_layers[idx].register_forward_hook(layer_input_hook(idx))
        hook_handles.append(handle)

    # 推理
    with torch.no_grad():
        output = model(**inputs)

    # 移除所有钩子，防止后续污染
    for h in hook_handles:
        h.remove()

    return layer_activations


def plt_llava_lang_activation(act_list, output_path, stat_axis=0):
    for idx, act in enumerate(act_list):
        if idx > 10 and idx < 30:
            continue
        plt_act_val_dim(act, f"{output_path}/layer_{idx}_input_act.png", title=f"Layer {idx} Input Activation", ylabel="Activation Value")


def plt_llava_vision_activation(act_list, output_path, stat_axis=0):
    for idx, act in enumerate(act_list):
        if idx > 10 and idx < 22:
            continue
        plt_act_val_dim(act, f"{output_path}/layer_{idx}_input_act.png", title=f"Layer {idx} Input Activation", ylabel="Activation Value")
    


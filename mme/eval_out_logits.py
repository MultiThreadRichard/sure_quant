import os
import time
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, LlavaForConditionalGeneration

from qwen_vl_utils import process_vision_info



# MME 测试数据集只有split='train'，获取测试集
def load_dataset_from_local(path):
    trainset = load_dataset('parquet', data_files=path, split='train')
    # testset = load_dataset('parquet', data_files=path, split='test')
    print(f'len(trainset): {len(trainset)}')
    # print(type(trainset))
    # print(f'len(testset): {len(testset)}')
    # print(type(testset))

    messages = []
    for item in trainset:
        # print(item)
        # break
        # mme_data = {
        #     'question_id': 'code_reasoning/0020.png',
        #     'image': Image.open('path_to_image/code_reasoning/0020.png'),  # 替换为实际路径
        #     'question': 'Is a python code shown in the picture? Please answer yes or no.',
        #     'answer': 'Yes',
        #     'category': 'code_reasoning'
        # }

        msg_item = [{
            "role": "user",
            "content": [
                {"type": "image", "image": item['image']},
                {"type": "text", "text": item['question']}
            ]
        }]
        messages.append(msg_item)

    return trainset, messages


def average_data_list(float_list):
    if len(float_list) == 0:
        return 0
    return sum(float_list) / len(float_list)


def preprocess_before_kl(fp_weight: torch.Tensor, q_weight: torch.Tensor):
    fp_weight = fp_weight.float()
    q_weight = q_weight.float()
    eval_len = min(len(fp_weight), len(q_weight))
    fp_weight = fp_weight[:eval_len]
    q_weight = q_weight[:eval_len]
    return fp_weight, q_weight



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
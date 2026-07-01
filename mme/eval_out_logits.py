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
    fp_weight: torch.Tensor,  # 全精度权重
    q_weight: torch.Tensor,   # 量化后权重
    fig_id: str = "",
    bins: int = 256,          # 直方图分箱数(bin)
    eps: float = 1e-10,        # 防止 log(0)
    paint: bool = False,
) -> float:
    """
    计算模型权重 全精度分布 P 与 量化分布 Q 之间的 KL 散度
    fp_weight 和 q_weight 形状相同
    """
    # output_path = f'/home/ecnu01/workspace/awq_learn/eval_test_sample/' + fig_id
    # if paint:
    #     os.makedirs(output_path, exist_ok=True)

    # save_fig_path1 = f'{output_path}/fp_flat.png'
    # save_fig_path2 = f'{output_path}/q_flat.png'
    # save_fig_path3 = f'{output_path}/fp_hist.png'
    # save_fig_path4 = f'{output_path}/q_hist.png'
    # save_fig_path5 = f'{output_path}/norm_p.png'
    # save_fig_path6 = f'{output_path}/norm_q.png'
    # save_fig_path7 = f'{output_path}/clamp_p.png'
    # save_fig_path8 = f'{output_path}/clamp_q.png'

    # 1. 展平权重 (权重矩阵是多维的，展平成一维计算)
    fp_flat = fp_weight.to(torch.float32).detach().cpu().flatten()
    q_flat = q_weight.to(torch.float32).detach().cpu().flatten()

    # if paint:
    #     plt_hist(fp_flat, save_fig_path1)
    #     plt_hist(q_flat, save_fig_path2)

    # 2. 统一取值范围（必须用相同的 min/max 分箱，否则 KL 无意义）
    min_val = min(fp_flat.min(), q_flat.min())
    max_val = max(fp_flat.max(), q_flat.max())

    # 3. 把一个一维张量里的数字，分成若干区间，统计每个区间有多少个数，返回每个区间的数量
    fp_hist = torch.histc(fp_flat, bins=bins, min=min_val, max=max_val)
    q_hist = torch.histc(q_flat, bins=bins, min=min_val, max=max_val)

    # if paint:
    #     plt_distribution_frequency(fp_hist, save_fig_path3)
    #     plt_distribution_frequency(q_hist, save_fig_path4)

    # 4. 转化为频率分布，norm到[0,1] 防止后续KL计算发生nan
    p = fp_hist / (fp_hist.sum())
    q = q_hist / (q_hist.sum())

    # if paint:
    #     plt_distribution_frequency(p, save_fig_path5)
    #     plt_distribution_frequency(q, save_fig_path6)

    # 5. 数值安全保护
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)

    # if paint:
    #     plt_distribution_frequency(p, save_fig_path7)
    #     plt_distribution_frequency(q, save_fig_path8)

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


def get_compressed_model(model_path, processor):
    # Load model. MXFP4 要求全精度加载 torch_dtype=torch.float32
    # 在 float16 下，指数位的表示范围较窄，容易在这些中间计算步骤中产生溢出或严重的舍入误差。
    # model_path = "/home/ecnu01/workspace/models/llava-1.5-7b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, device_map='auto', torch_dtype=torch.float32
    )
    model.eval()
    

    # Oneshot arguments
    # DATASET_ID = "flickr30k"

    DATASET_SPLIT = "test"
    NUM_CALIBRATION_SAMPLES = 128
    # NUM_CALIBRATION_SAMPLES = 512
    MAX_SEQUENCE_LENGTH = 2048


    # 加载本地数据集
    data_path_list = [
        '/home/ccwan/stu_Jiangtp/data/flickr30k/test-00000-of-00009.parquet',
        # '/home/ecnu01/workspace/data/flickr30k/data/test-00000-of-00009.parquet',
        # '/home/ecnu01/workspace/data/flickr30k/data/test-00001-of-00009.parquet',
        # '/home/ecnu01/workspace/data/flickr30k/data/test-00002-of-00009.parquet',
        # '/home/ecnu01/workspace/data/flickr30k/data/test-00003-of-00009.parquet',
        # '/home/ecnu01/workspace/data/flickr30k/data/test-00004-of-00009.parquet',
        # '/home/ecnu01/workspace/data/flickr30k/data/test-00005-of-00009.parquet',
        # '/home/ecnu01/workspace/data/flickr30k/data/test-00006-of-00009.parquet',
        # '/home/ecnu01/workspace/data/flickr30k/data/test-00007-of-00009.parquet',
        # '/home/ecnu01/workspace/data/flickr30k/data/test-00008-of-00009.parquet',
    ]
    # for data_path in data_path_list:
    #     dataset, messages = load_dataset_from_local(data_path)
    # dataset, messages = load_dataset_from_local(data_path_list[0])

    def preprocess_fn(example):
        # 根据你提供的特征结构：'image' 是图片对象, 'caption' 是文本列表或字符串
        image = example["image"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please describe this image\n"},
                    {"type": "image"},
                ],
            },
        ]
        
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt")

        # # Flickr30k 的 caption 通常是列表，取第一个即可
        # text = example["caption"][0] if isinstance(example["caption"], list) else example["caption"]
        
        # # 构造 LLaVA 标准提示词格式
        # prompt = f"USER: <image>\n{text}\nASSISTANT:"
        
        # # 处理成模型输入格式
        # inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)

        return {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "attention_mask": inputs["attention_mask"],
        }

    calib_dataset = load_dataset('parquet', data_files=data_path_list[0], split='train')
    # print(f'calib_dataset.column_names: {calib_dataset.column_names}')
    # pre_func(calib_dataset[0])

    # 转换为校准格式
    calib_dataset = calib_dataset.map(preprocess_fn, remove_columns=calib_dataset.column_names)
    calib_dataset.set_format(type="torch", columns=["input_ids", "pixel_values", "attention_mask"])

    # 只取前 128 个样本
    calib_dataset = calib_dataset.select(range(NUM_CALIBRATION_SAMPLES))

    print(f">>>>>>>> load dataset path: {data_path_list[0]}")
    print(f'len(calib_dataset): {len(calib_dataset)}')
    print(type(calib_dataset))

    # for item in calib_dataset:
    #     # print(item.keys())
    #     print(type(item['input_ids']))
    #     print(type(item['pixel_values']))
    #     print(type(item['attention_mask']))

    #     print(item['input_ids'].shape)
    #     print(item['pixel_values'].shape)
    #     print(item['attention_mask'].shape)

    #     break



    # Select quantization algorithm. In this case, we:
    #   * apply SmoothQuant to make the activations easier to quantize
    #   * quantize the weights to int8 with GPTQ (static per channel)
    #   * quantize the activations to int8 (dynamic per token)
    # recipe = [
    #     SmoothQuantModifier(
    #         smoothing_strength=0.5,
    #         mappings = [
    #             # Smooth the inputs going into the query, key, value projections of self-attention
    #             [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
    #             # Smooth the inputs going into the first feed-forward block (fc1)
    #             [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"]
    #         ],
    #         ignore=["re:.*lm_head", "re:.*vision_tower.*", "re:.*multi_modal_projector.*"],
    #     ),
    # ]

    # original Recipe
    # recipe = [
    #     GPTQModifier(
    #         targets="Linear",
    #         scheme="W4A16",
    #         ignore=["re:.*lm_head", "re:.*vision_tower.*", "re:.*multi_modal_projector.*"],
    #     ),
    # ]

    # ---------------------- FP8 量化配置 ----------------------
    # FP8_DYNAMIC: 动态 FP8 量化，激活值在运行时计算缩放因子，无需校准
    # recipe = QuantizationModifier(
    #     targets="Linear",          # 仅量化 Linear 层
    #     scheme="FP8_DYNAMIC",      # FP8 动态方案
    #     ignore=["re:.*lm_head", "re:.*vision_tower.*", "re:.*multi_modal_projector.*"],
    #     # kv_cache_scheme="FP8"      # 可选：KV 缓存也量化为 FP8
    #     # kv_cache_scheme={
    #     #     "num_bits": 8,
    #     #     "type": "float",
    #     #     "strategy": "tensor",  # 或者根据需要设为 "token"
    #     #     "symmetric": True
    #     # }
    # )

    # 静态 FP8 量化，需校准数据集
    # recipe = QuantizationModifier(
    #     targets="Linear",          # 仅量化 Linear 层
    #     scheme="FP8",      # FP8 动态方案
    #     ignore=["re:.*lm_head", "re:.*vision_tower.*", "re:.*multi_modal_projector.*"],
    # )


    # FP8_BLOCK
    # recipe = QuantizationModifier(
    #     targets="Linear",
    #     scheme="FP8_BLOCK",
    #     ignore=["re:.*lm_head", "re:.*vision_tower.*", "re:.*multi_modal_projector.*"],
    # )

    # nvfp4
    # recipe = QuantizationModifier(
    #     targets="Linear",
    #     scheme="NVFP4",  # NVIDIA 4位浮点格式
    #     ignore=["re:.*lm_head", "re:.*vision_tower.*", "re:.*multi_modal_projector.*"],
    # )

    # MXFP4
    # recipe = QuantizationModifier(
    #     targets="Linear",
    #     scheme="MXFP4",  # 实验性 MXFP4 格式
    #     ignore=["re:.*lm_head", "re:.*vision_tower.*", "re:.*multi_modal_projector.*"],
    # )

    # MXFP4A16
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="MXFP4A16",
        ignore=["re:.*lm_head", "re:.*vision_tower.*", "re:.*multi_modal_projector.*"],
    )

    SAVE_DIR = "/home/ccwan/stu_Jiangtp/model_repo/llava-1.5-7b-mxfp4"
    os.makedirs(SAVE_DIR, exist_ok=True)


    # model = model.to(device="cpu", dtype=torch.float32)
    torch.cuda.empty_cache()

    # Perform oneshot
    oneshot(
        model=model,
        tokenizer=model_path,
        dataset=None,
        # splits={"calibration": f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]"},
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        trust_remote_code_model=True,
        sequential_targets=["LlamaDecoderLayer"],
        output_dir=None,
    )
    print(">>>>>>>> oneshot done")

    # print(type(model))
    # print(model)
    print(f"Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

    return model



def eval_kl_full_qt():
    # TO MOD
    data_path_list = [
        '/home/ccwan/stu_Jiangtp/data/MME/data/test-00000-of-00004-a25dbe3b44c4fda6.parquet',
        '/home/ccwan/stu_Jiangtp/data/MME/data/test-00001-of-00004-7d22c7f1aba6fca4.parquet',
        '/home/ccwan/stu_Jiangtp/data/MME/data/test-00002-of-00004-594798fd3f5b029c.parquet',
        '/home/ccwan/stu_Jiangtp/data/MME/data/test-00003-of-00004-53ae1794f93b1e35.parquet',
    ]

    # TO MOD
    checkpoint = "/home/ccwan/stu_Jiangtp/model_repo/llava-7b-hf"

    # original model
    model = LlavaForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float16).eval()
    processor = AutoProcessor.from_pretrained(checkpoint)

    # quantized model
    qmodel = get_compressed_model(checkpoint, processor)

    # print(type(model))
    # print(model.config.model_type)

    print('>>>>>>>>>>>>> load model done.')

    # TO MOD
    output_path = '/home/ccwan/stu_Jiangtp/llm-compress-learn/mme_logits_eval'
    os.makedirs(output_path, exist_ok=True)

    output_filename = f'out_logits_kl_full_qt.txt'

    turn = 0
    score_list = []

    t_benchmark_start = time.time()
    for data_path in data_path_list:
        t_data, messages = load_dataset_from_local(data_path)
        print(f'>>>>>>>>> load {data_path}')
        # break

        print('>>>>>>>>> start eval')
        mode = 'a'
        with open(os.path.join(output_path, output_filename), mode, encoding="utf-8") as fout:
            unk = 0
            for item, msg_item in tqdm(zip(t_data, messages)):
                # torch.cuda.empty_cache()

                # 使用 processor 处理输入
                text = processor.apply_chat_template(msg_item, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(msg_item)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to("cuda")
                
                # print(inputs)
                # print(type(inputs)) # <class 'transformers.feature_extraction_utils.BatchFeature'>
                # print(inputs.keys())
                # print(f"inputs['input_ids'].shape: {inputs['input_ids'].shape}")
                # print(f"inputs['attention_mask'].shape: {inputs['attention_mask'].shape}")
                # print(f"inputs['pixel_values'].shape: {inputs['pixel_values'].shape}")
                # print(f"inputs['image_grid_thw'].shape: {inputs['image_grid_thw'].shape}")

                generated_ids = model.generate(**inputs, max_new_tokens=256)
                # print(generated_ids)
                # print(f"generated_ids.shape: {generated_ids.shape}")
                # print(f"generated_ids.dtype: {generated_ids.dtype}")

                q_gen_ids = qmodel.generate(**inputs, max_new_tokens=256)
                # print(q_gen_ids)
                # print(f"generated_ids.shape: {q_gen_ids.shape}")
                # print(f"generated_ids.dtype: {q_gen_ids.dtype}")

                logits_full, logits_quant = preprocess_before_kl(generated_ids[0], q_gen_ids[0])
                print(f"logits_full shape: {logits_full.shape}")
                print(f"logits_quant shape: {logits_quant.shape}")

                min_val = min(logits_full.min(), logits_quant.min())
                max_val = max(logits_full.max(), logits_quant.max())
                print("min_val: ", min_val)
                print("max_val: ", max_val)

                score = compute_kl_for_quantization(logits_full, logits_quant)

                # print(score, file=fout)

                score_list.append(score)

                # response = processor.batch_decode(
                #     generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                # )
                # print("Generated Response: ", response)

                # qresp = processor.batch_decode(
                #     q_gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                # )
                # print("qresp: ", qresp)

                # unk += 1
                # if unk == 5:
                #     break

                break
        

        print(f'>>>>>>>>> end eval')
        torch.cuda.empty_cache()
        turn += 1
        break


    t_benchmark_end = time.time()
    avg = average_data_list(score_list)
    print(f'>>>>>>>>> complete turn: {turn}')
    print(f'>>>>>>>>> total elapsed time: {t_benchmark_end - t_benchmark_start} s')
    print(f'>>>>>>>>> average score: {avg}')
    with open(os.path.join(output_path, output_filename), 'a', encoding="utf-8") as fout:
        fout.write(f'>>>>>>>>> average score: {avg}')


def eval_cos_full_qt():
    # TO MOD
    data_path_list = [
        '/home/ccwan/stu_Jiangtp/data/MME/data/test-00000-of-00004-a25dbe3b44c4fda6.parquet',
        '/home/ccwan/stu_Jiangtp/data/MME/data/test-00001-of-00004-7d22c7f1aba6fca4.parquet',
        '/home/ccwan/stu_Jiangtp/data/MME/data/test-00002-of-00004-594798fd3f5b029c.parquet',
        '/home/ccwan/stu_Jiangtp/data/MME/data/test-00003-of-00004-53ae1794f93b1e35.parquet',
    ]

    # TO MOD
    checkpoint = "/home/ccwan/stu_Jiangtp/model_repo/llava-7b-hf"

    # original model
    model = LlavaForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float16).eval()
    processor = AutoProcessor.from_pretrained(checkpoint)

    # quantized model
    qmodel = get_compressed_model(checkpoint, processor)

    # print(type(model))
    # print(model.config.model_type)

    print('>>>>>>>>>>>>> load model done.')

    # TO MOD
    output_path = '/home/ccwan/stu_Jiangtp/llm-compress-learn/mme_logits_eval'
    os.makedirs(output_path, exist_ok=True)

    output_filename = f'out_logits_cos_full_qt.txt'

    turn = 0
    score_list = []

    t_benchmark_start = time.time()
    for data_path in data_path_list:
        t_data, messages = load_dataset_from_local(data_path)
        print(f'>>>>>>>>> load {data_path}')
        # break

        print('>>>>>>>>> start eval')
        mode = 'a'
        with open(os.path.join(output_path, output_filename), mode, encoding="utf-8") as fout:
            for item, msg_item in tqdm(zip(t_data, messages)):
                # torch.cuda.empty_cache()

                # 使用 processor 处理输入
                text = processor.apply_chat_template(msg_item, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(msg_item)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to("cuda")
                
                # print(inputs)
                # print(type(inputs)) # <class 'transformers.feature_extraction_utils.BatchFeature'>
                # print(inputs.keys())
                # print(f"inputs['input_ids'].shape: {inputs['input_ids'].shape}")
                # print(f"inputs['attention_mask'].shape: {inputs['attention_mask'].shape}")
                # print(f"inputs['pixel_values'].shape: {inputs['pixel_values'].shape}")
                # print(f"inputs['image_grid_thw'].shape: {inputs['image_grid_thw'].shape}")

                generated_ids = model.generate(**inputs, max_new_tokens=256)
                # print(generated_ids)
                # print(f"generated_ids.shape: {generated_ids.shape}")
                # print(f"generated_ids.dtype: {generated_ids.dtype}")

                q_gen_ids = qmodel.generate(**inputs, max_new_tokens=256)
                # print(q_gen_ids)
                # print(f"generated_ids.shape: {q_gen_ids.shape}")
                # print(f"generated_ids.dtype: {q_gen_ids.dtype}")

                score = compute_cos_similarity(generated_ids[0], q_gen_ids[0])

                print(score, file=fout)

                score_list.append(score)

                # response = processor.batch_decode(
                #     generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                # )
                # print("Generated Response: ", response)

                # qresp = processor.batch_decode(
                #     q_gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                # )
                # print("qresp: ", qresp)

                # break
        

        print(f'>>>>>>>>> end eval')
        torch.cuda.empty_cache()
        turn += 1
        break


    t_benchmark_end = time.time()
    avg = average_data_list(score_list)
    print(f'>>>>>>>>> complete turn: {turn}')
    print(f'>>>>>>>>> total elapsed time: {t_benchmark_end - t_benchmark_start} s')
    print(f'>>>>>>>>> average score: {avg}')
    with open(os.path.join(output_path, output_filename), 'a', encoding="utf-8") as fout:
        fout.write(f'>>>>>>>>> average score: {avg}')



def eval_pearson_full_qt():
    # TO MOD
    data_path_list = [
        '/home/ccwan/stu_Jiangtp/data/MME/data/test-00000-of-00004-a25dbe3b44c4fda6.parquet',
        '/home/ccwan/stu_Jiangtp/data/MME/data/test-00001-of-00004-7d22c7f1aba6fca4.parquet',
        '/home/ccwan/stu_Jiangtp/data/MME/data/test-00002-of-00004-594798fd3f5b029c.parquet',
        '/home/ccwan/stu_Jiangtp/data/MME/data/test-00003-of-00004-53ae1794f93b1e35.parquet',
    ]

    # TO MOD
    checkpoint = "/home/ccwan/stu_Jiangtp/model_repo/llava-7b-hf"

    # original model
    model = LlavaForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float16).eval()
    processor = AutoProcessor.from_pretrained(checkpoint)

    # quantized model
    qmodel = get_compressed_model(checkpoint, processor)

    # print(type(model))
    # print(model.config.model_type)

    print('>>>>>>>>>>>>> load model done.')

    # TO MOD
    output_path = '/home/ccwan/stu_Jiangtp/llm-compress-learn/mme_logits_eval'
    os.makedirs(output_path, exist_ok=True)

    output_filename = f'out_logits_pearson_full_qt.txt'

    turn = 0
    score_list = []

    t_benchmark_start = time.time()
    for data_path in data_path_list:
        t_data, messages = load_dataset_from_local(data_path)
        print(f'>>>>>>>>> load {data_path}')
        # break

        print('>>>>>>>>> start eval')
        mode = 'a'
        with open(os.path.join(output_path, output_filename), mode, encoding="utf-8") as fout:
            for item, msg_item in tqdm(zip(t_data, messages)):
                # torch.cuda.empty_cache()

                # 使用 processor 处理输入
                text = processor.apply_chat_template(msg_item, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(msg_item)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to("cuda")
                
                # print(inputs)
                # print(type(inputs)) # <class 'transformers.feature_extraction_utils.BatchFeature'>
                # print(inputs.keys())
                # print(f"inputs['input_ids'].shape: {inputs['input_ids'].shape}")
                # print(f"inputs['attention_mask'].shape: {inputs['attention_mask'].shape}")
                # print(f"inputs['pixel_values'].shape: {inputs['pixel_values'].shape}")
                # print(f"inputs['image_grid_thw'].shape: {inputs['image_grid_thw'].shape}")

                generated_ids = model.generate(**inputs, max_new_tokens=256)
                # print(generated_ids)
                # print(f"generated_ids.shape: {generated_ids.shape}")
                # print(f"generated_ids.dtype: {generated_ids.dtype}")

                q_gen_ids = qmodel.generate(**inputs, max_new_tokens=256)
                # print(q_gen_ids)
                # print(f"generated_ids.shape: {q_gen_ids.shape}")
                # print(f"generated_ids.dtype: {q_gen_ids.dtype}")

                score = compute_pearson_correlation(generated_ids[0], q_gen_ids[0])

                print(score, file=fout)

                score_list.append(score)

                # response = processor.batch_decode(
                #     generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                # )
                # print("Generated Response: ", response)

                # qresp = processor.batch_decode(
                #     q_gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                # )
                # print("qresp: ", qresp)

                # break
        

        print(f'>>>>>>>>> end eval')
        torch.cuda.empty_cache()
        turn += 1
        break


    t_benchmark_end = time.time()
    avg = average_data_list(score_list)
    print(f'>>>>>>>>> complete turn: {turn}')
    print(f'>>>>>>>>> total elapsed time: {t_benchmark_end - t_benchmark_start} s')
    print(f'>>>>>>>>> average score: {avg}')
    with open(os.path.join(output_path, output_filename), 'a', encoding="utf-8") as fout:
        fout.write(f'>>>>>>>>> average score: {avg}')




# main
# eval_kl_full_qt()
# eval_cos_full_qt()
# eval_pearson_full_qt()









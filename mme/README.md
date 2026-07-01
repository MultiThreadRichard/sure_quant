# 1.下载模型、数据集
模型：https://huggingface.co/llava-hf/llava-1.5-7b-hf <br>
数据集：https://huggingface.co/datasets/lmms-lab/MME <br>

# 2. py环境
pip install -r requirements.txt <br>

# 3.安装fast-hadamard-transform
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git<br>
cd fast-hadamard-transform<br>

cmake 编译生成so<br>
python setup.py build_ext --inplace<br>

将编译后的fast-hadamard-transform 安装到conda环境<br>
pip install -e . --no-build-isolation<br>

# 4.执行入口
quant_llava.py

示例命令：
···
CUDA_VISIBLE_DEVICES=0 python quant_llava.py \
--model_name /home/ccwan/stu_Jiangtp/model_repo/llava-7b-hf \
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
--visual_split \
--visual_w_rtn \
--llm_w_rtn \
--w_asym
···
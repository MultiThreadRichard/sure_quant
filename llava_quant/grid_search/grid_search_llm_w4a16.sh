# LLM-W4A16: only LLM decoder weights quantized to int4, activations and other modules stay fp16
nohup python /home/ecnu03/workspace/sure_quant/llava_quant/grid_search/llava_quant_calib_wa_grid_search.py \
  --no-quantize-vision \
  --no-quantize-mm-proj \
  --quantize-language \
  --quantize-weight \
  --no-quantize-activation \
  --output-dir /home/ecnu03/workspace/sure_quant/runs/w4a16_language_only \
  > /home/ecnu03/workspace/sure_quant/llava_quant/grid_search/grid_search_llm_w4a16.log 2>&1 &



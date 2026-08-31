#nohup python /home/ecnu03/workspace/sure_quant/scripts/llava_quant_calib.py  > /home/ecnu03/workspace/sure_quant/scripts/log.outo 2>1 &

# W4A16: only LLM decoder weights quantized to int4, activations and other modules stay fp16
nohup python /home/ecnu03/workspace/sure_quant/scripts/llava_quant_calib_wa_grid_search.py \
  --no-quantize-vision \
  --no-quantize-mm-proj \
  --quantize-language \
  --quantize-weight \
  --no-quantize-activation \
  --output-dir /home/ecnu03/workspace/sure_quant/runs/w4a16_language_only \
  > /home/ecnu03/workspace/sure_quant/scripts/grid_search.log 2>&1 &



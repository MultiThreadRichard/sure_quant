nohup python /home/ecnu03/workspace/sure_quant/llava_quant/grid_search/llava_quant_calib_wa_grid_search.py \
  --quantize-vision \
  --quantize-mm-proj \
  --quantize-language \
  --quantize-weight \
  --no-quantize-activation \
  --output-dir /home/ecnu03/workspace/sure_quant/runs/all_w4a16 \
  > /home/ecnu03/workspace/sure_quant/llava_quant/grid_search/grid_search_all_w4a16.log 2>&1 &


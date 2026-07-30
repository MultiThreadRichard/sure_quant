import os
from datasets import load_dataset


"""
save flickr image
"""

def flickr_save_fig():
    data_path_list = [
        '/home/ecnu01/workspace/data/flickr30k/data/test-00000-of-00009.parquet',
    ]
    calib_sample_num = 10

    calib_dataset = load_dataset('parquet', data_files=data_path_list[0], split='train')
    print(f">>>>>>>> load dataset path: {data_path_list[0]}")

    calib_dataset = calib_dataset.select(range(calib_sample_num))
    print(f'len(calib_dataset): {len(calib_dataset)}')

    # 创建保存目录
    save_dir = '/home/ecnu01/workspace/sure_quant/logs/flick_figs'
    os.makedirs(save_dir, exist_ok=True)

    for idx, item in enumerate(calib_dataset):
        print(item['caption'])
        raw_image = item['image']

        # 保存图片到指定目录
        image_path = os.path.join(save_dir, f'flickr_image_{idx}.jpg')
        raw_image.save(image_path)
        print(f'Saved image {idx} to {image_path}')


flickr_save_fig()
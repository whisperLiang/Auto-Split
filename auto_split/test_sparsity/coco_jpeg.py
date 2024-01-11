
import os
import time

import numpy as np
from PIL import Image


if __name__ == "__main__":

    coco_val = '/data/datasets/coco2017'
    coco_val_dir =  '/data/datasets/coco2017/val2017'
    qf_list = [60,80] # 20,40,
    for qf in qf_list:
        print('Running qf: {}'.format(qf))
        output_dir = os.path.join(coco_val,'val2017_{}'.format(qf))
        image_list = os.listdir(coco_val_dir)
        for image in image_list:
            img_path = os.path.join(coco_val_dir,image)
            img = Image.open(img_path).convert('RGB')
            out_img_path = os.path.join(output_dir,image)
            img.save(out_img_path, "JPEG", quality=qf)

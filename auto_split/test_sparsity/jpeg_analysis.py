
import os
import time

import numpy as np
from PIL import Image


if __name__ == "__main__":

    solution = 3

    if solution == 1: # load a single image
        CODE_BASE_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
        img_path = os.path.join(CODE_BASE_DIR_PATH, "000000426773.jpg")
        img_path_out = os.path.join(CODE_BASE_DIR_PATH, "000000426773_out.jpg")

        start_time = time.time()
        img = Image.open(img_path).convert('RGB')
        img = img.resize((416, 416))
        print(np.asarray(img).shape)
        for i in range(1000):
            img.save(img_path_out, "JPEG", quality=100)

        duration = time.time() - start_time
        print("Time take: {} ms".format(duration))


    if solution == 2: # load to memory first
        input_img_dir = "/data/OpenImages_Classification/train/train0"
        CODE_BASE_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
        img_path_out = os.path.join(CODE_BASE_DIR_PATH, "000000426773_out.jpg")

        img_names = os.listdir(input_img_dir)
        imgs = []
        for i in range(100):
            img_path = os.path.join(input_img_dir, img_names[i])
            img = Image.open(img_path).convert('RGB')
            img = img.resize((416, 416))
            print("Image {} shape: {}".format(i, np.asarray(img).shape))
            imgs.append(img)

        print("Loaded {} images to memory".format(len(imgs)))
        start_time = time.time()
        for i in range(len(imgs)):
            imgs[i].save(img_path_out, "JPEG", quality=100)

        duration = time.time() - start_time
        print("Time take: {} ms".format(duration))


    if solution == 3: # load 1000 images and loop over
        input_img_dir = "/data/<path to >/datasets/coco2017/train2017"
        CODE_BASE_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
        output_path = os.path.join(CODE_BASE_DIR_PATH,'output')

        img_names = os.listdir(input_img_dir)
        duration = 0
        for i in range(500):
            img_path = os.path.join(input_img_dir, img_names[i])
            img = Image.open(img_path).convert('RGB')
            img = img.resize((416, 416))
            # print("Image {} shape: {}".format(i, np.asarray(img).shape))
            img_path_out = os.path.join(output_path, img_names[i])
            start_time = time.time()
            img.save(img_path_out, "JPEG", quality=100)
            duration += time.time() - start_time
        print("Time take: {} sec".format(duration/500))

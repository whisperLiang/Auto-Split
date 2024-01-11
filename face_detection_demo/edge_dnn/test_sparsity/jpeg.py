
import os
import time

import numpy as np
from PIL import Image
import cv2
import pathlib


def preprocess(image):
    input_shape = (288,512)
    image = cv2.imread(image)
    '''
    resize image with unchanged aspect ratio using padding by opencv
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    input_h, input_w = input_shape
    scale = min(float(input_w) / float(w), float(input_h) / float(h))
    nw = int(w * scale)
    nh = int(h * scale)

    image = cv2.resize(image, (nw, nh))

    new_image = np.zeros((input_h, input_w, 3), np.float32)
    new_image.fill(128)
    bh, bw, _ = new_image.shape
    # new_image[:nh, :nw, :] = image
    new_image[int((bh - nh) / 2):(nh + int((bh - nh) / 2)), int((bw - nw) / 2):(nw + int((bw - nw) / 2)),
    :] = image

    new_image /= 255.
    new_image = np.expand_dims(new_image, 0)
    new_image = Image.fromarray(new_image[0], mode='RGB')
    return new_image

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
            # img = img.resize((416, 416))
            print("Image {} shape: {}".format(i, np.asarray(img).shape))
            imgs.append(img)

        print("Loaded {} images to memory".format(len(imgs)))
        start_time = time.time()
        for i in range(len(imgs)):
            imgs[i].save(img_path_out, "JPEG", quality=100)

        duration = time.time() - start_time
        print("Time take: {} ms".format(duration))


    if solution == 3: # load 1000 images and loop over
        dir_name = "../test_images/"
        CODE_BASE_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
        output_path = os.path.join(CODE_BASE_DIR_PATH,'output_images')

        # img_names = os.listdir(input_img_dir)
        duration = 0
        image_list = os.listdir(dir_name)
        compression_results = {}
        for image_file in image_list:
            suffix = pathlib.Path(image_file).suffix
            if suffix != '.jpg':
                continue

            compression_results[image_file] = {}
            for quality in range(20,101,20):
        #     img_data = cv2.imread(os.path.join(dir_name, image_file))
                img_path = os.path.join(dir_name, image_file)
                # img_data = Image.open(img_path).convert('RGB')
                # img = img_data.resize((416, 416))
                img = preprocess(img_path)
                # print("Image {} shape: {}".format(i, np.asarray(img).shape))

                file_name = '{}_{}{}'.format(image_file.split(suffix)[0],quality,suffix)
                img_path_out = os.path.join(output_path, file_name)
                start_time = time.time()
                img.save(img_path_out, "JPEG", quality=quality)
                compression_results[image_file][quality] = round(os.path.getsize(img_path_out)/1024,2)
                duration += time.time() - start_time

        # Print csv results
        print('Image,20,40,60,80,100')

        for image, row in compression_results.items():
            print('{},{},{},{},{},{}'.format(image,row[20],row[40],row[60],row[80],row[100]))

        print("Time take: {} sec".format(duration/500))

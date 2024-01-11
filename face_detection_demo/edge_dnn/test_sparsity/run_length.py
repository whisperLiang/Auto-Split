# Fast run length encoding
import numpy as np
from PIL import Image

# def rle(img):
#     flat_img = img.flatten()
#     flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)
#
#     starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
#     ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
#     starts_ix = np.where(starts)[0] + 2
#     ends_ix = np.where(ends)[0] + 2
#     lengths = ends_ix - starts_ix
#
#     return starts_ix, lengths

from itertools import groupby
def encode_list(s_list):
    return [[len(list(group)), key] for key, group in groupby(s_list)]
n_list = [1,1,2,3,4,4.3,5, 1]
print("Original list:")
print(n_list)
print("\nList reflecting the run-length encoding from the said list:")
print(encode_list(n_list))
n_list = 'automatically'
print("\nOriginal String:")
print(n_list)
print("\nList reflecting the run-length encoding from the said string:")
print(encode_list(n_list))


# mask = np.arange(10)
# # mask = np.array(Image.open('../test_images/crowd_2.jpg'), dtype=np.uint8)
# mask_rle = rle(mask)
# print(mask_rle)


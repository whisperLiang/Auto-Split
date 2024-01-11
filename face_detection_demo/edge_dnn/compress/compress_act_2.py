
import numpy as np
# from bitstruct import pack
from bitstring import BitArray
import bitarray
from random import seed
import time
import math

def compress(input, H, W, C_by_2, num_bits):
    input_packed = np.zeros([H, W, C_by_2], dtype=np.uint8)
    for i in range(0,C_by_2):
        x = np.left_shift(input[:, :, 2*i], num_bits)
        y = input[:,:,2*i+1]
        input_packed[:,:,i] = x+y
    return input_packed

# Unpack
# H,W,C, num_bits
def decompress(input_packed, H,W,C,C_by_2, num_bits):
    recovered_input = np.zeros([H, W, C], dtype=np.uint8)
    for i in range(0,C_by_2):
        l_x = np.left_shift(input_packed[:, :, i],num_bits)
        y = np.right_shift(l_x, num_bits)
        x = np.right_shift(input_packed[:, :, i],num_bits)
        recovered_input[:, :, 2*i] = x
        recovered_input[:, :, 2*i+1] = y

    return recovered_input



if __name__ == '__main__':
    np.random.seed(1)
    # B=1
    # H = 2
    # W = 3
    # C = 4
    # C_by_2 = 2
    B=1
    H = 36
    W = 64
    C = 256
    C_by_2 = 128

    shape = (B,H,W,C)
    vol = np.prod(shape)
    num_bits = 4

    #--------------
    # Ex 1:
    #--------------
    input = np.random.randint(0, 15, size=vol, dtype=np.uint8)
    # input = np.arange(0, vol, dtype=np.uint8)
    input.resize((H,W,C))
    time1 = time.time()
    input_packed = compress(input, H, W, C_by_2, num_bits)
    recovered_input = decompress(input_packed, H,W,C,C_by_2, num_bits)
    time2 = time.time()
    time_diff = time2 - time1
    diff = (recovered_input - input).sum()
    print('diff 1: {}, time: {}'.format(diff, time_diff))

    # #--------------
    # # Ex 2
    # #--------------
    # input_q = np.random.randint(0, 15, size=vol, dtype=np.uint8)
    # input_q.resize(shape)
    # time2 = time.time()
    # input_q_shape = input_q.shape
    # (_, H, W, C) = input_q.shape
    # input_q.resize(H, W, C)
    # C_by_2 = int(math.ceil(C / 2))
    # input_q = input_q.astype(np.uint8)
    # input_packed_q = compress(input_q, H, W, C_by_2, num_bits)
    # xx = input_packed_q.flatten().astype(int).tolist()
    # time3 = time.time()
    # print(' compression_time: {}'.format(time3 - time2))
    #
    #
    # (b, H, W, C) = input_q_shape
    # C_by_2 = int(math.ceil(C / 2))
    # input_packed = np.array(xx, dtype=np.uint8)
    # input_packed.resize((H, W, C_by_2))
    # recovered_input_q = np.zeros(input_q_shape, dtype=np.uint8)
    # recovered_input_q = decompress(input_packed, H, W, C, C_by_2)
    # recovered_input_q.resize(input_q_shape)
    # recovered_input_q = recovered_input_q.astype(float)
    # diff = (input_q - recovered_input_q).sum()
    # print('diff: {}'.format(diff))
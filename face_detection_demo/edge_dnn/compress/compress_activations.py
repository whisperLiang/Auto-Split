import numpy as np
import math
import sys
import time


def twos_complement(n, bits):
    s = bin(n & int("1"*bits, 2))[2:]
    return ("{0:0>%s}" % (bits)).format(s)

def invert_twos_complement(input_value, num_bits):
    if len(input_value)!= num_bits:
        return int(input_value,2)
    else:
        x = int(input_value[1:],2)
        max = 2 ** (num_bits-1)
        sign = input_value[0]

        if sign == '1':
            orig_val = x - max
        else:
            orig_val = x

    return orig_val

def pack_data(input_list, num_bits, min_val, max_val, available_bits):
    len_inputs = len(input_list)
    concat_val_list = []
    num_vals_per_int_list = []
    is_last_iter = False
    for idx in range(0,len_inputs, available_bits):
        start_idx = idx
        end_idx = idx + available_bits
        if len_inputs <= end_idx:
            end_idx = len_inputs
            is_last_iter = True

        concat_val_str = ''
        count = 0
        for val in input_list[start_idx:end_idx]:
            assert val >= min_val and val <= max_val, 'values are out of bit range. val:{}, ' \
                    'bit range:[{},{}]'.format(val,min_val, max_val)
            bin_val = twos_complement(val,num_bits)
            # print(bin_val)
            concat_val_str += bin_val
            count +=1
        num_vals_per_int_list.append(count)

        concat_val = int(concat_val_str, 2)
        concat_val_list.append(concat_val)
        # For all packed INT Numvres -- #count=15 or num_vals_per_int except the last INT;
        # Hence, we need to tell how many are there as a protocol for the last one.
        if is_last_iter:
            concat_val_list.append(count)
    return concat_val_list

def unpack_data(concat_val_list, num_bits, num_vals_per_int):
    orig_vals = []
    count_list = []
    len_val_str_list = []
    #NOTE: as per the protocol, the last concat_val tells the number of values in previous concat_val.
    # It is not a real value.
    num_vals_in_last_iter = concat_val_list[-1]
    concat_val_list = concat_val_list[:-1]
    len_concat_val_list = len(concat_val_list)
    for idx, concat_val in enumerate(concat_val_list):

        if idx == len_concat_val_list -1 :
            num_vals_in_this_iter = num_vals_in_last_iter
        else:
            num_vals_in_this_iter = num_vals_per_int

        # print(concat_val)
        bin_str = bin(concat_val)
        assert bin_str[0:3] != '-0b', '-ve number {}'.format(bin_str)
        val_str = bin_str[2:]
        split_str_list= []
        count = 0
        len_val_str = len(val_str)
        len_val_str_list.append(len_val_str)
        for i in range(len_val_str,0,-1*num_bits):
            end_idx = i
            padding = 0
            if (i - num_bits) >= 0:
                start_idx = i - num_bits
            else:
                start_idx = 0
                padding = -1*(i - num_bits)

            count +=1
            x = val_str[start_idx:end_idx]
            split_str_list.insert(0,x)

        if count < num_vals_in_this_iter:
            # Remaining numbers were zeros and was ignored while converting back in python.
            # TODO: Append zeros until count is reached.
            remaining_vals = num_vals_in_this_iter - count
            for j in range(0,remaining_vals):
                split_str_list.insert(0,'0000')
                count +=1

        count_list.append(count)
        orig_vals += [invert_twos_complement(x,num_bits) for x in split_str_list]
    return orig_vals

if __name__ == '__main__':
    int_size = sys.maxsize
    available_bits = int(math.log2(int_size))
    num_bits = 4
    num_vals_per_int = math.floor(available_bits/num_bits)
    np.random.seed(1)
    input_np = np.random.randint(-7,8,size=(36,64,256))
    input_np_shape = input_np.shape
    input_list = list(input_np.flatten())
    len_input_list = len(input_list)
    # print('Original inputs: {}'.format(input_list))
    print('Original inputs size: {}'.format(input_np_shape))
    max_val = 2**(num_bits-1)-1
    min_val = -1*(2**(num_bits-1))

    time1 = time.time()
    concat_val_list = pack_data(input_list, num_bits, min_val, max_val, num_vals_per_int)
    # print(concat_val_list)
    len_act_transmission = len(concat_val_list)
    print('Transmission activation numbers:{} bytes:{}'.format(len_act_transmission, len_act_transmission*8))
    recovered_val_list = unpack_data(concat_val_list, num_bits, num_vals_per_int)
    time2 = time.time()
    time_diff = time2 - time1

    len_recovered_list = len(recovered_val_list)
    assert len_recovered_list == len_input_list, 'orig_len={}, recovered_len={}'.format(
        len_input_list, len_recovered_list)

    recovered_np = np.array(recovered_val_list)
    recovered_np  = recovered_np.reshape(input_np_shape)
    # print('recovered inputs: {}'.format(recovered_val_list))
    diff = input_np - recovered_np
    print('diff : {}, time: {}'.format(diff.sum(), time_diff))


import xmlrpc.client
import base64

# #-----------------------------------------------------
# # Example 0
# #-----------------------------------------------------
# device = xmlrpc.client.ServerProxy("http://localhost:1234/RPC2")
# # x = base64.urlsafe_b64encode('Some String'.encode('UTF-8')).decode('ascii')
# # text = base64.urlsafe_b64encode('Some String'.encode('UTF-8'))
# # y = text.decode('ascii')
# # device.output(y)
#
# #-----------------------------------------------------
# # Example 1:
# #-----------------------------------------------------
# # Edge side
# # First encode a binary data as a string in utf-8 format.
# text = base64.b64encode(b"this is a test string asdlka jlkdjasd")
# # device.output(text)
#
# # -- cloud side
# xx_bytes = base64.b64decode(text)
# recover_text = xx_bytes.decode("utf-8")
# print(recover_text)
#
# #-----------------------------------------------------
# # Example 2:
# #-----------------------------------------------------
#
# import sys
# import struct
#
# from bitstring import BitArray
# import bitarray
#
#
# edge_byte_order = sys.byteorder
# symbols = [-4, -3, -2, -1, 0, 1, 2, 3, 3]
# binary_result = '100101110111000001010011011'
#
# # 1. Convert array/list to Bitarray
# x = BitArray().join(BitArray(int=x, length=3) for x in symbols)
#
# x_bin_str = str(x.bin)
# if x_bin_str == binary_result:
#     print('BitArray is correct')
# else:
#     print('BitArray is wrong')
# print('BitArray: {}'.format(x.bin))
#
# # 2. Bitarray to bytes
# x_bytes = x.tobytes()
# print('BitArray to bytes: {} \nEncoding ...'.format(x_bytes))
# # 3. bytes to -- utf-8 encoding --> to transfer the string over RPC..
# xx_encoded = base64.b64encode(x_bytes)
# print('Encoded  BitArray bytes: {}'.format(xx_encoded))
#
# # Cloud side ----
#
# # C1. decode utf-8 string to bytes
# xx_bytes = base64.b64decode(xx_encoded)
# print('Decoding ...\nDecoded  BitArray bytes: {}'.format(xx_bytes))
# ba = bitarray.bitarray()
# ba.frombytes(xx_bytes) # Reads 8 bit binary
# print('Decoded  BitArrays 1: {}'.format(ba))
#
# # C2. decode bytes to array/list
# from bitstruct import unpack
# dtype= 's{}'.format(3)*9
# recovered_symbols = unpack(dtype, xx_bytes)
# print(recovered_symbols)

#-----------------------------------------------------
# Example 3: Minimalist code from Example 2
#-----------------------------------------------------

import sys
import struct

from bitstring import BitArray
import bitarray

def method1():
    edge_byte_order = sys.byteorder
    symbols = [0,1,2,3,4,6,7]
    len_symbols = len(symbols)
    num_bits=3
    # 1. Convert array/list to Bitarray
    x = BitArray().join(BitArray(uint=x, length=num_bits) for x in symbols)
    # 2. Bitarray to bytes
    x_bytes = x.tobytes()
    # 3. bytes to -- utf-8 encoding --> to transfer the string over RPC..
    xx_encoded = base64.b64encode(x_bytes)
    # # text = base64.urlsafe_b64encode('Some String'.encode('UTF-8'))
    yy = xx_encoded.decode('ascii')

    device = xmlrpc.client.ServerProxy("http://localhost:1234/RPC2")
    # device.output(yy)

    # Cloud side ----
    # C1. decode utf-8 string to bytes
    xx_bytes = base64.b64decode(yy)
    # C2. decode bytes to array/list
    from bitstruct import unpack
    dtype= 'u{}'.format(num_bits)*len_symbols
    recovered_symbols = unpack(dtype, xx_bytes)
    print(recovered_symbols)


def method2():
    edge_byte_order = sys.byteorder
    symbols = [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15]
    len_symbols = len(symbols)
    num_bits=4
    # 1. Convert array/list to Bitarray
    x = BitArray().join(BitArray(uint=x, length=num_bits) for x in symbols)
    hex_str = x.tobytes().hex()
    # device = xmlrpc.client.ServerProxy("http://localhost:1234/RPC2")
    # device.output(yy)

    # Cloud side ----
    from bitstruct import unpack
    dtype= 'u{}'.format(num_bits)*len_symbols
    xx_bytes = bytes.fromhex(hex_str)
    recovered_symbols = unpack(dtype, xx_bytes)
    print(recovered_symbols)

if __name__ == '__main__':
    method2()

#---------------------------------------------------------------------------
# xx = xx_bytes[0]
# num_bits = xx.bit_length()
# reqd_bits = 4
# bin(151>>4) # Convert to signed.
#
# ba2 = BitArray(int=xx, length=3)
#
# # x_int = bin(int.from_bytes(xx_bytes, byteorder=edge_byte_order, signed=True))
# # print('Decoded  BitArrays 2: {}'.format(x_int))
# # text = base64.urlsafe_b64encode(x.encode('UTF-8'))
# zz = struct.unpack('hh', xx_bytes) # Minimum representation is 8 bit= short in C or 'h'.
# (1024).to_bytes(2, byteorder='big')
# print(y)
# # x.tobytes()
# # print(x)



# from bitarray import bitarray
# d = bitarray('0' * 30, endian='little')
# d[5] = 1
# print(struct.unpack("<L", d)[0])
# d[6] = 1
# print(struct.unpack("<L", d)[0])
#
#
#
# import bitarray
# ba = bitarray.bitarray()
# ba.frombytes('Hi'.encode('utf-8'))


















# #-------------------------------------
# # Ex 1: Working bits --> bytes and back
# #-------------------------------------
#
# from bitstruct import *
#
# symbols = [-4, -3, -2, -1, 0, 1, 2, 3, 3]
# x = pack('u1u3u4s16', 1, 2, 3, -4)
# print(x)
# y = unpack('u1u3u4s16', x)
# print(y)
# s = calcsize('u1u3u4s16')
# print(s)
#
#
# dtype = 's3'*len(symbols)
# p_x = pack(dtype,symbols[0],symbols[1],symbols[2],symbols[3],
#            symbols[4],symbols[5],symbols[6],symbols[7], symbols[8])
# print(p_x)
# p_y = unpack(dtype, p_x)
# print(p_y)
# print(calcsize(dtype))
# #-------------------------------------
#-------------------------------------
# Ex 2:
#-------------------------------------

import base64

# # CLient ---
# message = 'Python is fun'
# base64_bytes = base64.urlsafe_b64encode(message.encode('UTF-8'))
#
# # Server --
# message_bytes = base64.b64decode(base64_bytes)
# message = message_bytes.decode('ascii')
# print(message)

#-------------------------------------
# Ex 3:
#-------------------------------------
# import base64
# encode = base64.b64encode(b"this is a test string asdlka jlkdjasd")
# decode = base64.b64decode(encode)
#
# print(encode, end="\n")
# print(decode)


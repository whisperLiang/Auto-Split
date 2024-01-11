from attrdict import AttrDict
import random
import math
import numpy as np

# --------------------------------------
# Functions to quantize
# --------------------------------------
class Quantize():
	# def __init__(self):
	# 	# print('Symmetric Quantization')
	# # TODO: add scale to quantization
	def linear_quantize_params(self, num_bits, sat_min, sat_max):
		# Given bits one needs to know the best K - to minimize MSE"
		# K= (2^num_bits - 1);
		# delta = (wmax - wmin)/K; => delta = 1/scale
		# i=floor(0.5 + x/delta);
		# Q(x) = delta*i
		# sat_max = pow(2,num_bits)-1
		# sat_min = -1* sat_max + 1
		diff = sat_max - sat_min
		scale = (pow(2,num_bits) - 1)/diff
		zero_point = 0
		return scale, zero_point

	def linear_quantize(self, input, scale, zero_point):
		# input_q = np.around(scale * input - zero_point)
		input_q = np.around(scale.reshape(-1, 1) * input - zero_point)
		return input_q
	def linear_dequantize(self, input_q, scale, zero_point):
		x_discrete = (input_q + zero_point)/scale.reshape(-1, 1)
		return x_discrete

	def quantize(self, input, num_bits, sat_min=-1, sat_max=1, scale=None, zero_point=None, clipping=None):

		if scale is None or zero_point or None or clipping is None:
			scale, zero_point = self.linear_quantize_params(num_bits, sat_min, sat_max)
			input_q = self.linear_quantize(input, scale, zero_point)
			input_discrete = self.linear_dequantize(input_q, scale, zero_point)
		else:
			# TODO: Add clipping logic. Clip vectors beyong clipping point.
			#  for act: (0,clipping), for weights = currently not supported.
			# input = input > clipping
			scale_arr = np.full_like(num_bits, scale[0], dtype='float32')
			input_q = self.linear_quantize(input, scale_arr, zero_point)
			input_discrete = self.linear_dequantize(input_q, scale, zero_point)

		return input_discrete
# --------------------------------------






# if __name__ == '__main__':
# 	# parser = argparse.ArgumentParser()
# 	# args = parser.parse_args()
# 	all_weights_names = load_all_weights_name()
#
# 	# fake DNN to simulate additivity and quantization
# 	DNN_D = AttrDict()
# 	DNN_D = create_fake_dnn(all_weights_names,DNN_D)
#
# 	# dnn state to be used for bit search
# 	dnn_state_D = AttrDict()
# 	dnn_state_D = create_model_dict(all_weights_names, dnn_state_D)
# 	# dnn_state_D = add_lambda_init(dnn_state_D,DNN_D)
#
# 	print(all_weights_names[0:5])


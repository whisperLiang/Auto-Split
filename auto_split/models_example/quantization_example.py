import torch

# 1. Get a quantized Tensor by quantizing unquantized float Tensors
float_tensor = torch.randn(2, 2, 3)

scale, zero_point = 1e-4, 2
dtype = torch.qint32
q_per_tensor = torch.quantize_per_tensor(float_tensor, scale, zero_point, dtype)

# we also support per channel quantization
scales = torch.tensor([1e-1, 1e-2, 1e-3])
zero_points = torch.tensor([-1, 0, 1])
channel_axis = 2
q_per_channel = torch.quantize_per_channel(float_tensor, scales, zero_points, axis=channel_axis, dtype=dtype)

# 2. Create a quantized Tensor directly from empty_quantized functions
# Note that _empty_affine_quantized is a private API, we will replace it
# something like torch.empty_quantized_tensor(sizes, quantizer) in the future
q = torch._empty_affine_quantized([10], scale=scale, zero_point=zero_point, dtype=dtype)

# 3. Create a quantized Tensor by assembling int Tensors and quantization parameters
# Note that _per_tensor_affine_qtensor is a private API, we will replace it with
# something like torch.form_tensor(int_tensor, quantizer) in the future
int_tensor = torch.randint(0, 100, size=(10,), dtype=torch.uint8)

# The data type will be torch.quint8, which is the corresponding type
# of torch.uint8, we have following correspondance between torch int types and
# torch quantized int types:
# - torch.uint8 -> torch.quint8
# - torch.int8 -> torch.qint8
# - torch.int32 -> torch.qint32
q = torch._make_per_tensor_quantized_tensor(int_tensor, scale, zero_point)  # Note no `dtype`


## Operations on quantized tensor

# Dequantize
q_made_per_tensor = q
dequantized_tensor = q_made_per_tensor.dequantize()

# Quantized Tensor supports slicing like usual Tensors do
s = q_made_per_tensor[2] # a quantized Tensor of with same scale and zero_point
                         # that contains the values of the 2nd row of the original quantized Tensor
                         # same as q_made_per_tensor[2, :]

# Assignment
q_made_per_tensor[0] = 3.5 # quantize 3.5 and store the int value in quantized Tensor

# Copy
# we can copy from a quantized Tensor of the same size and dtype
# but different scale and zero_point
scale1, zero_point1 = 1e-1, 0
scale2, zero_point2 = 1, -1
q1 = torch._empty_affine_quantized([2, 3], scale=scale1, zero_point=zero_point1, dtype=torch.qint8)
q2 = torch._empty_affine_quantized([2, 3], scale=scale2, zero_point=zero_point2, dtype=torch.qint8)
q2.copy_(q1)

# Permutation
q1.transpose(0, 1)  # see https://pytorch.org/docs/stable/torch.html#torch.transpose
q1.permute([1, 0])  # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute
q1.contiguous()  # Convert to contiguous Tensor

# Serialization and Deserialization
import tempfile
with tempfile.NamedTemporaryFile() as f:
    torch.save(q2, f)
    f.seek(0)
    q3 = torch.load(f)

## INSPECTING A QUANTIZED TENSOR

# Check size of Tensor
q.numel(), q.size()

# Check whether the tensor is quantized
q.is_quantized

# Get the scale of the quantized Tensor, only works for affine quantized tensor
q.q_scale()

# Get the zero_point of quantized Tensor
q.q_zero_point()

# get the underlying integer representation of the quantized Tensor
# int_repr() returns a Tensor of the corresponding data type of the quantized data type
# e.g.for quint8 Tensor it returns a uint8 Tensor while preserving the MemoryFormat when possible
q.int_repr()

# If a quantized Tensor is a scalar we can print the value:
# item() will dequantize the current tensor and return a Scalar of float
q[0].item()

# printing
print(q)
# tensor([0.0026, 0.0059, 0.0063, 0.0084, 0.0009, 0.0020, 0.0094, 0.0019, 0.0079,
#         0.0094], size=(10,), dtype=torch.quint8,
#        quantization_scheme=torch.per_tensor_affine, scale=0.0001, zero_point=2)

# indexing
print(q[0]) # q[0] is a quantized Tensor with one value
# tensor(0.0026, size=(), dtype=torch.quint8,
#        quantization_scheme=torch.per_tensor_affine, scale=0.0001, zero_point=2)
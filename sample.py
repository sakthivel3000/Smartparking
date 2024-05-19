import torch

# Get the floating point info for float32
float_info = torch.finfo(torch.float32)

# Print out the properties of float32
print("dtype:", float_info.dtype)
print("bits:", float_info.bits)
print("eps:", float_info.eps)
# print("epsneg:", float_info.epsneg)
print("max:", float_info.max)
print("min:", float_info.min)
print("tiny:", float_info.tiny)
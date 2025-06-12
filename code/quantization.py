import torch

def quantize(input_data, scale):
  return torch.round(input_data * scale).to(torch.int8)

def dequantize(quantized_data, scale):
  return quantized_data.to(torch.float32) / scale

input_data = torch.Tensor([1.1, -0.3, 16.3, 1.2])
scale = 127.0 / torch.max(torch.abs(input_data))

quantized_data = quantize(input_data, scale)
print("量化结果：", quantized_data)

dequantized_data = dequantize(quantized_data, scale)
print("反量化结果", dequantized_data)
print("误差:", dequantized_data-input_data)
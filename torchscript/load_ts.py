"""
加载 torchscript 模型
"""
import torch
model = torch.jit.load('./ts_model/lenet_trace.ts')
input = torch.rand(1, 1, 32, 32)
pred = model(input)
print(pred.size())
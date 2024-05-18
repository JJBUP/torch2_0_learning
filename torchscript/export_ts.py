"""
导出TorchScript
"""
import torch
import model.lenet as lenet

model = lenet.LeNet()
input_data = torch.randn(1, 1, 32, 32)

# trace 模型导出
traced_script_module = torch.jit.trace(model, input_data)
traced_script_module.save("../ts_model/lenet_trace.ts")
print("trace 模型导出成功")

# script 模型导出
script_module = torch.jit.script(model)
script_module.save("../ts_model/lenet_script.ts")
print("script 模型导出成功")
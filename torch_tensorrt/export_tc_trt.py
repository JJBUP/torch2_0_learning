# 使用torch_compile作为IR导出
import os
import torch_tensorrt
import torch
import torchvision as tv
import numpy as np
from PIL import Image

torch_tensorrt.runtime.set_multi_device_safe_mode(True)  # 启用多设备安全模式
torch_tensorrt.set_device(1)
device = torch.device("cuda:1")
# 模型设置
# model = lenet.LeNet().eval()  # trt模型导出必须为测试模型
model = tv.models.resnet34(weights=tv.models.ResNet34_Weights.IMAGENET1K_V1).eval()
model.half()  # 模型与输入输入类型一致
model.to(device)  # 输入数据与模型均要在cuda上
# 设置输入数据
img = Image.open("../data/test.jpg").resize((224, 224)).convert("RGB")
input_exp = (
    torch.tensor(np.asarray(img), dtype=torch.float16).reshape(1, 3, 224, 224).to(device)
)
# 模型执行
pred: torch.Tensor = model(input_exp)
pred_cls = pred.argmax(-1).cpu().detach()
print("模型预测结果:")
print(pred_cls)

# 模型编译
inputs = [
    torch_tensorrt.Input(
        (1, 3, 224, 224),
        dtype=torch.float16,
        torch_tensor=torch.randn((1, 3, 224, 224), dtype=torch.float16).to(device),
    )
]
enabled_precisions = {
    torch.float16,
}  # Run with fp16 (torch model to trt)
trt_dy_module = torch_tensorrt.compile(
    module=model,
    ir="torch_compile",
    inputs=inputs,
    enabled_precisions=enabled_precisions,
)

pred = trt_dy_module(input_exp)
pred_cls = pred[0].argmax(-1).cpu().detach()
print("预测结果:")
print(pred_cls)



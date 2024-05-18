"""
针对四种IR进行速度测试
使用torch_tensorrt加载模型进行推理
"""

import time
import torch_tensorrt
import torch
import torchvision as tv
import PIL.Image as Image
import numpy as np

torch_tensorrt.set_device(1)
device = torch.device("cuda:1")
ephocs = 100
start_ephocs = 1
test_ephocs = ephocs - start_ephocs


def time_fn(fn, inputs: torch.Tensor):
    start_time = time.time()
    result = fn(inputs)
    end_time = time.time()
    return result, end_time - start_time


# 测试数据
img = Image.open("../data/test.jpg").resize((224, 224)).convert("RGB")
input_exp = torch.tensor(np.asarray(img), dtype=torch.float16).reshape(1, 3, 224, 224)
input_exp = input_exp.to(device)

model = (
    tv.models.resnet34(weights=tv.models.ResNet34_Weights.IMAGENET1K_V1)
    .to("cuda")
    .half()
    .eval()
)
print(">>>>>>>>>>torch.nn.model<<<<<<<<<<<<<")
time_sum = 0
for i in range(ephocs):
    result, time_span = time_fn(model, input_exp)
    # print(f"Inference time: {time_span} seconds. Prediction: {result.argmax(-1)}")
    if i > start_ephocs-1:
        time_sum = time_sum + time_span
print(f"Average inference time: {time_sum / test_ephocs} seconds")
benchmark_time = time_sum / test_ephocs
# %%

print(">>>>>>>>>>trt_model_ts<<<<<<<<<<<<<")
# torchscript IR
trt_model_ts = torch.jit.load("../trt_model/resnet34.trt.ts", map_location=device)
# print(trt_model_ts.code)
# print(trt_model_ts.graph)
time_sum = 0
for i in range(ephocs):
    result, time_span = time_fn(trt_model_ts, input_exp)
    # print(f"Inference time: {time_span} seconds. Prediction: {result.argmax(-1)}")
    if i > start_ephocs-1:
        time_sum = time_sum + time_span
print(
    f"Average inference time: {time_sum / test_ephocs} seconds, Lifting ratio: {(benchmark_time/(time_sum / test_ephocs))}"
)

print(">>>>>>>>>>trt_model_dyn_ts<<<<<<<<<<<<<")
# dynamo IR
trt_model_dyn_ts = torch.jit.load(
    "../trt_model/resnet34_dynamo.trt.ts", map_location=device
)
# print(trt_model_dyn_ts.code)
# print(trt_model_dyn_ts.graph)
time_sum = 0
for i in range(ephocs):
    result, time_span = time_fn(trt_model_dyn_ts, input_exp)
    # print(f"Inference time: {time_span} seconds. Prediction: {result.argmax(-1)}")
    if i > start_ephocs-1:
        time_sum = time_sum + time_span
print(
    f"Average inference time: {time_sum / test_ephocs} seconds, Lifting ratio: {(benchmark_time/(time_sum / test_ephocs))}"
)

print(">>>>>>>>>>trt_model_dyn_exp<<<<<<<<<<<<<")
trt_model_dyn_exp = torch.export.load("../trt_model/resnet34_dynamo.trt.exp")
# trt_model_dyn_exp.module().to(device)
# print(trt_model_dyn_ts.graph)
time_sum = 0
for i in range(ephocs):
    result, time_span = time_fn(trt_model_dyn_exp, input_exp)
    # print(f"Inference time: {time_span} seconds. Prediction: {result[0].argmax(-1)}")
    if i > start_ephocs-1:
        time_sum = time_sum + time_span
print(
    f"Average inference time: {time_sum / test_ephocs} seconds, Lifting ratio: {(benchmark_time/(time_sum / test_ephocs))}"
)

print(">>>>>>>>>>trt_model_tc<<<<<<<<<<<<<")
# torch_compile IR

inputs = [
    torch_tensorrt.Input(
        (1, 3, 224, 224),
        dtype=torch.float16,
        torch_tensor=torch.randn((1, 3, 224, 224), dtype=torch.float16).to(device),
    )
]
trt_model_tc = torch_tensorrt.compile(
    module=model, ir="torch_compile", inputs=inputs, enabled_precisions={torch.float16}
)
# print(trt_model_tc)
# print(trt_model_tc)
time_sum = 0
for i in range(ephocs):
    result, time_span = time_fn(trt_model_tc, input_exp)
    # print(f"Inference time: {time_span} seconds. Prediction: {result[0].argmax(-1)}")
    if i > start_ephocs-1:
        time_sum = time_sum + time_span
print(
    f"Average inference time: {time_sum / test_ephocs} seconds, Lifting ratio: {(benchmark_time/(time_sum / test_ephocs))}"
    )

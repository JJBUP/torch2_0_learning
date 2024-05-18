# %%
"""
使用装饰器编译函数
设置控制流，使得dynamo捕获多个子图
并使用自己的后端，输出dynamo 捕获的子图
"""
from typing import List
import torch
import torch._dynamo.comptime
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(">>> my_compiler() invoked:")
    print(">>> FX graph:")
    gm.graph.print_tabular()
    print(f">>> Code:\n{gm.code}")
    return gm.forward  # return a python callable


@torch.compile(backend=my_compiler)
def foo(x, y):
    x = x.sin()
    y = y.cos()
    if y.sum() < 0:
        return x + y
    else:
        return x - y


if __name__ == "__main__":
    a, b = torch.randn(10) + 1, torch.ones(10) + 1
    foo(a, b)
    a, b = torch.randn(10) - 1, torch.ones(10) - 1
    foo(a, b)
    pass 
# %%
"""
使用函数的方式编译函数
"""
from typing import List
import torch
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(">>> my_compiler() invoked:")
    print(">>> FX graph:")
    gm.graph.print_tabular()
    print(f">>> Code:\n{gm.code}")
    return gm.forward  # return a python callable

def foo(x, y):
    x = y / (torch.abs(y) + 1)
    if y.sum() < 0:
        y = y * -1
    return x * y


if __name__ == "__main__":
    a, b = torch.randn(10), torch.ones(10)
    foo_cp = torch.compile(foo, backend=my_compiler)
    foo_cp(a, b)


# %%
"""
编译模型与eager模式速度测试
"""
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))


mod = MyModule()
opt_mod = torch.compile(mod)
print(opt_mod(torch.randn(10, 100)))
# %%
import torch._dynamo

torch._dynamo.config.suppress_errors = True


# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


# Generates random input and targets data for the model, where `b` is
# batch size.
def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

N_ITERS = 10 # 迭代次数
from torchvision.models import densenet121
def init_model():
    return densenet121().to(torch.float32).cuda()

model = init_model()

# Reset since we are using a different mode.
import torch._dynamo
torch._dynamo.reset()
model_opt = torch.compile(model, mode="reduce-overhead")

inp = generate_data(16)[0]
with torch.no_grad():
    print("eager:", timed(lambda: model(inp))[1])
    print("compile:", timed(lambda: model_opt(inp))[1])
# %%
"""
使用explain()查看 TorchDynamo 在哪里破坏图表
"""
from typing import List
import torch

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(">>> my_compiler() invoked:")
    print(">>> FX graph:")
    gm.graph.print_tabular()
    print(f">>> Code:\n{gm.code}")
    return gm.forward  # return a python callable

def foo(x, y):
    x = y / (torch.abs(y) + 1)
    if y.sum() < 0:
        y = y * -1
    return x * y

if __name__ == "__main__":

    explain_output = torch._dynamo.explain(foo)(torch.randn(10) + 1, torch.ones(10) + 1)
    print(explain_output)
    explain_output = torch._dynamo.explain(foo)(torch.randn(10) - 1, torch.ones(10) - 1)
    print(explain_output)

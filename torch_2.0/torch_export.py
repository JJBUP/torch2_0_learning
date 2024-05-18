import torch
from torch.export import export
"""
export(
    f: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Dict[str, Dict[int, Dim]]] = None
) -> ExportedProgram
"""
# torch.export.export() 跟踪调用 f(*args, **kwargs) 的张量计算图，并将其包装在 ExportedProgram 中，可以序列化或稍后使用不同的输入执行。
# 虽然输出 ExportedGraph 是可调用的，并且可以按照与原始输入可调用相同的方式进行调用，但它不是 torch.nn.Module 。

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x, y):
        return torch.nn.functional.relu(self.lin(x + y), inplace=True)

mod = MyModule()
exported_mod = export(mod, (torch.randn(8, 100), torch.randn(8, 100)))
print(type(exported_mod))
print(exported_mod.module()(torch.randn(8, 100), torch.randn(8, 100)))

print(">>>exported_mod")
print(exported_mod)
# graph 属性是从我们导出的函数追踪到的 FX 图，即所有 PyTorch 操作的计算图。
print(">>>exported_mod.graph")
print(exported_mod.graph)

# graph_module 属性是包装 graph 属性的 GraphModule ，以便它可以作为 torch.nn.Module 运行。
print(">>>exported_mod.graph_module")
print(exported_mod.graph_module)
# graph_signature 属性是导出的函数的签名，即输入和输出张量的形状和类型，以及参数和缓冲区。
print(">>>exported_mod.graph_signature")
print(exported_mod.graph_signature)
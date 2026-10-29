"""Day 01：用来练习"怎么运行一个 .py 文件"的小例子。

运行方式（任选其一）：
  终端：conda activate dl && python day01/hello_torch.py
  VS Code：打开本文件，点右上角 ▶ 按钮
"""
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

x = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
print("x =\n", x)
print("形状:", x.shape, " 设备:", x.device)
print("x 的转置乘 x =\n", x.T @ x)

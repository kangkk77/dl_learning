"""检查深度学习环境：PyTorch 版本、可用设备、在设备上做一次矩阵乘法。"""
import platform
import sys
import time

import torch


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    print(f"Python      : {sys.version.split()[0]} ({platform.machine()})")
    print(f"PyTorch     : {torch.__version__}")
    print(f"CUDA 可用   : {torch.cuda.is_available()}")
    print(f"MPS 已编译  : {torch.backends.mps.is_built()}")
    print(f"MPS 可用    : {torch.backends.mps.is_available()}")

    device = pick_device()
    print(f"使用设备    : {device}")

    a = torch.randn(1000, 1000)
    b = torch.randn(1000, 1000)
    a_dev, b_dev = a.to(device), b.to(device)

    c_dev = a_dev @ b_dev  # 预热
    sync(device)
    start = time.perf_counter()
    c_dev = a_dev @ b_dev
    sync(device)
    elapsed = (time.perf_counter() - start) * 1000

    max_err = (c_dev.cpu() - a @ b).abs().max().item()
    print(f"矩阵乘法    : {tuple(c_dev.shape)} on {c_dev.device}, {elapsed:.2f} ms")
    print(f"与 CPU 最大误差: {max_err:.2e}")
    print("环境检查通过 ✅")


if __name__ == "__main__":
    main()

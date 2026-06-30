**注意：**Pytorch Quantization 的开发已转移至 [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)。建议所有开发者使用 TensorRT Model Optimizer，以获取量化与压缩方面的最新进展。Pytorch Quantization 的代码仍将保留可用，但不会再进一步开发。

# Pytorch Quantization

PyTorch-Quantization 是一个用于对 PyTorch 模型进行模拟量化训练与评估的工具包。量化可以自动或手动添加到模型中，从而允许针对精度和性能对模型进行调优。量化兼容 NVIDIA 的高性能整数内核，可充分利用 integer Tensor Cores。量化后的模型可导出为 ONNX，并由 TensorRT 8.0 及更高版本导入。

## 安装

#### 二进制包

```bash
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
```

#### 从源码安装

```bash
git clone https://github.com/NVIDIA/TensorRT.git
cd tools/pytorch-quantization
```

安装 PyTorch 及前置依赖
```bash
pip install -r requirements.txt
# CUDA 10.2 用户
pip install torch>=1.9.1
# CUDA 11.1 用户
pip install torch>=1.9.1+cu111
```

编译并安装 pytorch-quantization
```bash
# 要求 Python 版本 >= 3.7，GCC 版本 >= 5.4
python setup.py install
```

#### NGC 容器

`pytorch-quantization` 已预装在 NVIDIA NGC PyTorch 容器中，例如 `nvcr.io/nvidia/pytorch:22.12-py3`

## 资源

* Pytorch Quantization Toolkit [userguide](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html)
* Quantization Basics [whitepaper](https://arxiv.org/abs/2004.09602)
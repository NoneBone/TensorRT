# 将 FP32 ONNX 模型转换为混合 FP32-FP16 精度，用于 TensorRT Strong Typing

**目录**

- [Description](#description)

- [How does this sample work?](#how-does-this-sample-work)

  -   [Data preparation and the original ONNX model verification](#data-preparation-and-the-original-onnx-model-verification)

  -   [Model conversion and the converted ONNX model verification](#model-conversion-and-the-converted-onnx-model-verification)

  -   [Build and verify TensorRT engine from the converted ONNX model](#build-and-verify-tensorrt-engine-from-the-converted-onnx-model)

- [Prerequisites](#prerequisites)

- [Running the sample](#running-the-sample)

  -   [Sample `--help`options](#sample-help-options)

- [Additional resources](#additional-resources)

- [License](#license)

- [Changelog](#changelog)

- [Known issues](#known-issues)

## Description

本示例 `strongly_type_autocast`使用 ModelOpt 的 AutoCast 工具将 FP32 ONNX 模型转换为混合 FP32-FP16 精度，并以 TensorRT 的 strong typing 模式构建 engine 并运行推理。

[AutoCast](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/8_autocast.html)是一个用于将 FP32 ONNX 模型转换为混合精度（FP32-FP16 或 FP32-BF16）模型的工具。AutoCast 智能地选择需要保持在 FP32 精度的节点，以维持模型精度，同时让其余节点受益于低精度加速。AutoCast 会在被选中的节点周围自动插入 cast 操作。

[Strong Typing vs Weak Typing](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/capabilities.html#strong-vs-weak-typing)

对于 strong typing，TensorRT 严格遵循 ONNX 框架中的类型语义；对于 weak typing，若能提升性能，TensorRT 可能会为 tensor 替换不同的精度。Weak typing 已在 10.12 版本中被弃用。建议在进行 TensorRT strong typing 优化前，先使用 AutoCast 工具将 FP32 ONNX 模型转换为混合精度。

## How does this sample work?

本示例包含三个阶段：

- [Data preparation and the original ONNX model verification](#data-preparation-and-the-original-onnx-model-verification)

- [Model conversion and the converted ONNX model verification](#model-conversion-and-the-converted-onnx-model-verification)

- [Build and verify TensorRT engine from the converted ONNX model](#build-and-verify-tensorrt-engine-from-the-converted-onnx-model)

### Data preparation and the original ONNX model verification

原始输入数据为 pgm 格式，包含 0~9 共十张数字图片。输入数据需要转换为 npz 格式。

原始 ONNX 模型为 fp32 精度。为验证原始模型，使用 ONNX Runtime 运行推理，打印预测的数字，并将所有模型输出保存为黄金参考（gold references）。

### Model conversion and the converted ONNX model verification

可以使用 ModelOpt 的 AutoCast python 包将原始 fp32 ONNX 模型转换为 fp32-fp16 混合精度：

````
```
from modelopt.onnx.autocast import convert_to_mixed_precision

converted_model = convert_to_mixed_precision(
    onnx_path="model.onnx",
    low_precision_type="fp16",            # 或 "bf16"
    nodes_to_exclude=None,                # 可选，保持 FP32 的节点名称模式列表
    op_types_to_exclude=None,             # 可选，保持 FP32 的 op 类型列表
    data_max=512,                         # 可转换节点的 I/O 最大绝对值
    init_max=65504,                       # 可转换 initializer 的 I/O 最大绝对值
    keep_io_types=False,                  # 是否保留输入/输出类型
    calibration_data=None,                # 可选，输入数据文件路径
)
```
````

生成 fp32-fp16 混合精度模型后，使用 ONNX Runtime 在转换后的 ONNX 模型上运行推理，打印预测数字并将模型输出与黄金参考进行对比。

### Build and verify TensorRT engine from the converted ONNX model

从转换后的 ONNX 模型构建 TensorRT engine，通过 `NetworkDefinitionCreationFlag.STRONGLY_TYPED`启用 strong type：

````
```
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
```
````

在生成的 trt engine 上运行推理，打印预测数字并将模型输出与黄金参考进行对比。

## Prerequisites

本示例所需的依赖

1. 安装 Python 依赖：

   ```
   pip3 install --upgrade pip
   pip3 install -r requirements.txt
   ```

2. TensorRT

3. 如果使用 TensorRT container，MNIST 数据集位于 data 目录下（通常为 `/usr/src/tensorrt/data/mnist`），同时也随 [TensorRT tarball](https://developer.nvidia.com/nvidia-tensorrt-download)一同分发。

## Running the sample

1. 运行示例：

   `python3 sample.py [--mnist_dir] [--working_dir]`

2. 确认示例运行成功。若成功，应看到如下输出：

   ```
   Sample finished successfully.
   ```

### Sample --help options

查看完整的可用选项列表及说明，可使用 `-h`或 `--help`命令行参数。

# Additional resources

以下资源有助于更深入理解 AutoCast 与 TensorRT strong typing：

**文档**

- [Guide of TensorRT-Model-Optimizer Autocast](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/8_autocast.html)

- [TensorRT Strong Typing vs Weak Typing](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/capabilities.html#strong-vs-weak-typing)

- [Introduction To NVIDIA's TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)

- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)

- [NVIDIA's TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

关于使用、复制和分发的条款与条件，请参阅 [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)文档。

# Changelog

2025 年 9 月

此为 `README.md`文件的初始版本。

# Known issues

本示例暂无已知问题。
# 从 ONNX 模型构建与重构权重剥离引擎入门

**目录**

- [描述](#description)

- [本示例工作原理](#how-does-this-sample-work)

- [前置条件](#prerequisites)

- [运行示例](#running-the-sample)

  - [示例 `--help`选项](#sample-help-options)

- [额外资源](#additional-resources)

- [许可证](#license)

- [变更日志](#changelog)

- [已知问题](#known-issues)

## 描述

本示例 `sample_weight_stripping`是一个 Python 示例，演示如何使用 TensorRT 构建一个权重剥离的引擎，随后将其重构为完整的推理引擎。

## 本示例工作原理？

本示例演示了如何使用 TensorRT Python API 从 ONNX 模型文件构建一个权重剥离的引擎，从而减小保存的 engine 体积。随后，通过 parser refitter 以原始 ONNX 模型作为输入，对权重剥离的引擎进行重构。重构后的完整引擎用于推理，并能保证性能和精度无损。在本示例中，我们使用 ResNet50 来展示相关特性。

## 前置条件

1. 安装 Python 依赖项。

   ```
   pip3 install -r requirements.txt
   ```

2. 准备示例数据

   参见主示例 README 中的 [准备示例数据](https://yuanbao.tencent.com/README.md#preparing-sample-data)。

## 运行示例

1. 构建并保存普通引擎和权重剥离引擎：

   ```
   python3 build_engines.py --output_stripped_engine=stripped_engine.trt --output_normal_engine=normal_engine.trt
   ```

   运行此步骤后，你将看到两个保存好的 TensorRT 引擎。`stripped_engine.trt`包含一个剥离后的引擎（约 2.3MB），而 `normal_engine.trt`包含一个包含所有权重的普通引擎（约 51MB）。通过使用剥离引擎构建，我们可以大幅减小保存的 engine 文件体积。

   **注意：** 如果 TensorRT 示例数据未安装在默认位置（例如 `/usr/src/tensorrt/data/`），则必须指定模型目录。例如：`--stripped_onnx=/path/to/my/data/`设置构建权重剥离引擎的模型路径，`--original_onnx=/path/to/my/data/`设置构建普通引擎的模型路径。在大多数情况下，两者可以使用同一个 ONNX 模型。

2. 重构权重剥离引擎，并使用权重剥离引擎和普通引擎执行推理：

   ```
   python3 refit_engine_and_infer.py --stripped_engine=stripped_engine.trt -–normal_engine=normal_engine.trt
   ```

3. 验证示例是否成功运行。如果示例运行成功，你应该会看到类似以下的输出。重构后的剥离引擎的预测结果与普通引擎一致，且无性能损失。

   ```
   Normal engine inference time on 100 cases: 0.1066 seconds
   Refitted stripped engine inference time on 100 cases: 0.0606 seconds
   Normal engine correctly recognized data/samples/resnet50/tabby_tiger_cat.jpg as tiger cat
   Refitted stripped engine correctly recognized data/samples/resnet50/tabby_tiger_cat.jpg as tiger cat
   ```

### 示例 --help 选项

要查看可用选项的完整列表及其描述，请使用 `-h`或 `--help`命令行选项。

# 额外资源

以下资源有助于更深入地了解如何使用 Python 将模型导入 TensorRT：

**ResNet-50**

- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

**文档**

- [NVIDIA TensorRT 示例简介](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)

- [使用 Python API 操作 TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)

- [NVIDIA TensorRT 文档库](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# 许可证

关于使用、复制和分发的条款与条件，请参阅 [TensorRT 软件许可协议](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)文档。

# 变更日志

2025 年 10 月

迁移至强类型 API。

2025 年 8 月

移除对 < 3.10 Python 版本的支持。

2024 年 2 月

本示例首次发布。

# 已知问题

本示例暂无已知问题。
# 使用 Python 实现 IProgressMonitor 回调简介

 en [English](./README.md) ｜ zh_CN [简体中文](./README_zh.md)

**目录**

- [描述](#description)

- [本示例如何工作？](#how-does-this-sample-work)

  -   [simple_progress_monitor](#simple_progress_monitor)

- [前置条件](#prerequisites)

- [运行示例](#running-the-sample)

  -   [示例 `--help`选项](#sample-help-options)

- [额外资源](#additional-resources)

- [许可证](#license)

- [变更日志](#changelog)

- [已知问题](#known-issues)

## 描述

本示例 `simple_progress_monitor`是一个 Python 示例，它使用 TensorRT 及其内置的 ONNX 解析器，对以 ONNX 格式保存的 ResNet-50 模型执行推理。它在 TensorRT 构建引擎的过程中显示动态进度条。

## 本示例如何工作？

### simple_progress_monitor

本示例演示了如何使用开源 ONNX 解析器从 ONNX 模型文件构建引擎并运行推理。ONNX 解析器可与任何支持 ONNX 格式的框架（通常为 `.onnx`文件）配合使用。`IProgressMonitor`对象会接收构建进度的更新，并将其作为 ASCII 进度条显示在标准输出（stdout）上。

## 前置条件

1. 安装 Python 依赖项。

```
pip3 install -r requirements.txt
```

1. 准备示例数据

   请参阅主示例 README 中的[准备示例数据](#preparing-sample-data)。

## 运行示例

1. 从终端运行示例以创建 TensorRT 推理引擎并运行推理：

   `python3 simple_progress_monitor.py`

   **注意：** 如果 TensorRT 示例数据未安装在默认位置，则必须指定 `data`目录。例如：`python3 simple_progress_monitor.py -d $TRT_DATADIR`

   **注意：** 请勿将此脚本的输出重定向到文件或管道。

2. 验证示例是否成功运行。如果示例运行成功，您应看到类似于以下的输出：

   `Correctly recognized data/samples/resnet50/reflex_camera.jpeg as reflex camera`

### 示例 --help 选项

要查看可用选项的完整列表及其描述，请使用 `-h`或 `--help`命令行选项。例如：

```
usage: simple_progress_monitor.py [-h] [-d DATADIR]

Runs a ResNet50 network with a TensorRT inference engine. Displays intermediate build progress.

optional arguments:
 -h, --help            show this help message and exit
 -d DATADIR, --datadir DATADIR
                       Location of the TensorRT sample data directory.
                       (default: /usr/src/tensorrt/data)
```

# 额外资源

以下资源有助于更深入地了解如何使用 Python 将模型导入 TensorRT：

**ResNet-50**

- [用于图像识别的深度残差学习](https://arxiv.org/pdf/1512.03385.pdf)

**解析器**

- [ONNX 解析器](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/parsers/Onnx/pyOnnx.html)

**文档**

- [NVIDIA TensorRT 示例简介](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)

- [使用 Python API 操作 TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)

- [使用 Python 解析器导入模型](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_model_python)

- [NVIDIA TensorRT 文档库](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

**终端转义序列**

- Linux: [XTerm 控制序列](https://invisible-island.net/xterm/ctlseqs/ctlseqs.html)

- Windows: [控制台虚拟终端序列](https://learn.microsoft.com/en-us/windows/console/console-virtual-terminal-sequences)

# 许可证

有关使用、复制和分发的条款和条件，请参阅 [TensorRT 软件许可协议](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)文档。

# 变更日志

2025年10月

迁移至强类型 API。

2025年8月

移除对 Python < 3.10 版本的支持。

2023年8月

移除对 Python < 3.8 版本的支持。

2023年6月

创建并审核了本 `README.md`文件。

# 已知问题

本示例中暂无已知问题。
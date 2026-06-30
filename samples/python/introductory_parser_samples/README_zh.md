# 使用 Python 将 ONNX 模型导入 TensorRT 简介

**目录**

- [描述](#description)

- [本示例如何工作？](#how-does-this-sample-work)

  -   [onnx_resnet50](#onnx_resnet50)

- [前置条件](#prerequisites)

- [运行示例](#running-the-sample)

  -   [示例 `--help`选项](#sample-help-options)

- [额外资源](#additional-resources)

- [许可证](#license)

- [变更日志](#changelog)

- [已知问题](#known-issues)

## 描述

本示例 `introductory_parser_samples`是一个 Python 示例，演示如何使用 TensorRT 及其内置的 ONNX 解析器，对以 ONNX 格式保存的 ResNet-50 模型执行推理。

## 本示例如何工作？

### onnx_resnet50

本示例演示了如何使用开源 ONNX 解析器从 ONNX 模型文件构建 engine，然后运行推理。ONNX 解析器可与任何支持 ONNX 格式的框架（通常为 `.onnx`文件）配合使用。

## 前置条件

1. 安装 Python 依赖项。

```
pip3 install -r requirements.txt
```

1. 准备示例数据

请参阅主示例 README 中的[准备示例数据](https://yuanbao.tencent.com/README.md#preparing-sample-data)。

## 运行示例

1. 运行示例以创建 TensorRT 推理 engine 并运行推理：

   `python3 onnx_resnet50.py`

   **注意：** 如果 TensorRT 示例数据未安装在默认位置，则必须指定 `data`目录。例如：`python3 onnx_resnet50.py -d $TRT_DATADIR`

2. 验证示例是否成功运行。如果示例运行成功，您应看到类似于以下的输出：

   `Correctly recognized data/samples/resnet50/reflex_camera.jpeg as reflex camera`

### 示例 --help 选项

要查看可用选项的完整列表及其描述，请使用 `-h`或 `--help`命令行选项。例如：

```
usage: onnx_resnet50.py [-h] [-d DATADIR]

Runs a ResNet50 network with a TensorRT inference engine.

optional arguments:
 -h, --help            show this help message and exit
 -d DATADIR, --datadir DATADIR
                       Location of the TensorRT sample data directory.
                       (default: /usr/src/tensorrt/data)
```

# 额外资源

以下资源有助于更深入地了解如何使用 Python 将模型导入 TensorRT：

**ResNet-50**

- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

**解析器**

- [ONNX 解析器](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/parsers/Onnx/pyOnnx.html)

**文档**

- [NVIDIA TensorRT 示例简介](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)

- [使用 Python API 操作 TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)

- [在 Python 中使用解析器导入模型](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_model_python)

- [NVIDIA TensorRT 文档库](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# 许可证

有关使用、复制和分发的条款和条件，请参见 [TensorRT 软件许可协议](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)文档。

# 变更日志

2025 年 10 月

迁移至强类型 API。

2025 年 8 月

移除对 < 3.10 Python 版本的支持。

2023 年 8 月

移除对 < 3.8 Python 版本的支持。

2022 年 8 月

移除 Caffe 和 UFF 解析器相关选项。

2019 年 2 月

重新创建、更新并审核了本 `README.md`文件。

# 已知问题

本示例中暂无已知问题。
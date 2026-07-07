# ONNX 模型的 TensorRT Engine Refitting

 en [English](./README.md) ｜ zh_CN [简体中文](./README_zh.md)

**目录**

- [简介](#description)

- [本样例如何工作？](#how-does-this-sample-work)

- [前置条件](#prerequisites)

- [运行样例](#running-the-sample)

- [额外资源](#additional-resources)

- [许可证](#license)

- [变更日志](#changelog)

- [已知问题](#known-issues)

## 简介

本样例展示了如何通过解析器重构由 ONNX 模型refit engine。使用修改后的 [ONNX BiDAF 模型](https://github.com/onnx/models/tree/main/validated/text/machine_comprehension/bidirectional_attention_flow)作为样例模型，该模型实现了论文 [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603)中描述的双向注意力流（BiDAF）网络。

## 本样例如何工作？

本样例通过 ONNX-graphsurgeon（在 `prepare_model.py`中）替换原始 ONNX 模型中不支持的节点（HardMax / Compress），并构建一个可重构的 TensorRT engine。随后，在 `build_and_refit_engine.py`中，分别使用伪造权重和正确权重对该 engine 进行重构，并对样例上下文和查询语句执行推理。

## 前置条件

本样例所需的依赖项：

1. 安装 Python 依赖：

```
pip3 install -r requirements.txt
```

1. TensorRT

2. [ONNX-GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)

3. 下载样例数据。参见 [通用设置指南](README.md)中的“下载样例数据”部分。

## 运行样例

运行这些脚本时需要指定数据目录（通过 `-d /path/to/data`或环境变量 `TRT_DATA_DIR`）。若未指定，将抛出错误。以下示例采用 `TRT_DATA_DIR`方式。

- 准备 ONNX 模型。（需指定数据目录。）

  ```
  export TRT_DATA_DIR=/root/cys/PROJECT/00-COMMON/DEMO/02-TensorRT/data
  python3 prepare_model.py
  ```

输出应类似于以下内容：

```
Modifying the ONNX model ...
Modified ONNX model saved as bidaf-modified.onnx
Done.
```

该脚本会修改来自 [onnx/models](https://github.com/onnx/models/raw/c02f8c8699fc12273649e658b8d2a1a8e32a35d0/text/machine_comprehension/bidirectional_attention_flow/model/bidaf-9.onnx)的原始模型，并保存一个可由 TensorRT 解析和运行的 ONNX 模型。

原始 ONNX 模型包含四个 CategoryMapper 节点，用于将四个输入字符串数组映射为整数数组。由于 TensorRT 不支持字符串数据类型和 CategoryMapper 节点，我们将这四个节点对应的映射关系导出为 JSON 文件（`model/CategoryMapper_{4-6}.json`），并用它们来预处理输入数据。现在，这四个输入变成了原始 CategoryMapper 节点的四个输出。

同时，不支持的 HardMax 节点和 Compress 节点分别被替换为 ArgMax 节点和 Gather 节点。

- 构建 TensorRT engine、重构 engine 并运行推理。

  `python3 build_and_refit_engine.py --weights-location GPU`

该脚本将从修改后的 ONNX 模型构建 TensorRT engine，然后从 GPU 权重重构 engine，并对样例上下文和查询语句执行推理。

首次运行上述命令时，输出应类似于以下内容：

```
Loading ONNX file from path bidaf-modified.onnx...
Beginning ONNX file parsing
[09/25/2023-08:48:16] [TRT] [W] ModelImporter.cpp:407: Make sure input CategoryMapper_4 has Int64 binding.
[09/25/2023-08:48:16] [TRT] [W] ModelImporter.cpp:407: Make sure input CategoryMapper_5 has Int64 binding.
[09/25/2023-08:48:16] [TRT] [W] ModelImporter.cpp:407: Make sure input CategoryMapper_6 has Int64 binding.
[09/25/2023-08:48:16] [TRT] [W] ModelImporter.cpp:407: Make sure input CategoryMapper_7 has Int64 binding.
Completed parsing of ONNX file
Network inputs:
CategoryMapper_4 <class 'numpy.int64'> (-1, 1)
CategoryMapper_5 <class 'numpy.int64'> (-1, 1, 1, 16)
CategoryMapper_6 <class 'numpy.int64'> (-1, 1)
CategoryMapper_7 <class 'numpy.int64'> (-1, 1, 1, 16)
Building an engine from file bidaf-modified.onnx; this may take a while...
Completed creating Engine
Refitting engine from GPU weights...
Engine refitted in 39.88 ms.
Doing inference...
Doing inference...
Refitting engine from GPU weights...
Engine refitted in 0.27 ms.
Doing inference...
Doing inference...
Passed
```

注意，第二次重构会比第一次快得多。再次运行上述命令时，engine 将从 plan 文件反序列化，输出应类似于以下内容：

```
Reading engine from file bidaf.trt...
Refitting engine from GPU weights...
Engine refitted in 32.64 ms.
Doing inference...
Doing inference...
Refitting engine from GPU weights...
Engine refitted in 0.41 ms.
Doing inference...
Doing inference...
Passed
```

要从 CPU 权重重构 engine，将命令改为 `python3 build_and_refit_engine.py --weights-location CPU`。输出应类似于以下内容：

```
Reading engine from file bidaf.trt...
Refitting engine from CPU weights...
Engine refitted in 45.18 ms.
Doing inference...
Doing inference...
Refitting engine from CPU weights...
Engine refitted in 1.20 ms.
Doing inference...
Doing inference...
Passed
```

还有一个 `--version-compatible`选项用于启用 engine 版本兼容性。如果已安装，`tensorrt_dispatch`包将用于重构和运行版本兼容的 engine，而不是 `tensorrt`包。要构建和重构版本兼容的 engine，请运行命令 `python3 build_and_refit_engine.py --version-compatible`，输出应与上述情况类似。

# 额外资源

以下资源有助于更深入地了解本样例所使用的模型：

**模型**

- [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603)

**文档**

- [NVIDIA TensorRT 样例简介](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)

- [使用 Python API 操作 TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)

- [在 Python 中使用解析器导入模型](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_model_python)

- [NVIDIA TensorRT 文档库](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# 许可证

有关使用、复制和分发的条款与条件，请参见 [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)文档。

# 变更日志

2025 年 10 月

- 迁移至强类型 API。

2025 年 8 月：

- 移除对 < 3.10 Python 版本的支持。

2024 年 1 月：

- 新增对重构版本兼容 engine 的支持。

2023 年 8 月：

- 新增对从 GPU 权重重构 engine 的支持。

- 移除对 < 3.8 Python 版本的支持。

2020 年 10 月：本样例被重新创建、更新并审核。

# 已知问题

本样例暂无已知问题。
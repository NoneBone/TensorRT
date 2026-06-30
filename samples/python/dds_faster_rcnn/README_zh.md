# DDS Faster R-CNN 目标检测（TensorRT 实现）

 en [English](./README.md) ｜ zh_CN [简体中文](./README_zh.md)

## 简介

`dds_faster_rcnn`样例演示了如何在 TensorRT 中使用 [tensorrt.IOutputAllocator](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/ExecutionContext.html#tensorrt.IOutputAllocator)来执行带有数据依赖形状（DDS）输出的网络。本样例展示了一个端到端的工作流，用于构建和运行目标检测模型 [Faster-RCNN](https://arxiv.org/abs/1506.01497)。

### 什么是数据依赖形状（DDS）？

数据依赖形状（DDS）指的是神经网络中层输出的形状取决于该层的输入数据；换言之，仅通过检查层输入张量的形状无法推断其输出形状。一个典型例子是 `INonZeroLayer`的输出形状，它由输入张量中非零元素的数量决定。

DDS 输出在涉及动态处理的模型中十分常见，例如目标检测、图像分割和自然语言处理模型。

### 什么是 `IOutputAllocator`？

`IOutputAllocator`是 TensorRT 中的一个接口，用于定义一个负责动态分配和管理 TensorRT engine 输出张量设备内存的类。实现此接口的类必须提供分配和释放输出张量内存的方法，因为输出张量的大小可能随输入数据而变化。

### 为什么需要实现 `IOutputAllocator`？

在传统模型中，输出形状通常在构建期就是固定且已知的。然而，对于数据依赖形状（DDS）输出，输出大小只有在推理阶段才能确定。这意味着输出张量的内存分配无法在模型实际运行特定输入之前确定。为了应对这种情况，TensorRT 提供了 `IOutputAllocator`接口，允许开发者为 DDS 输出实现自定义的内存分配策略。通过实现此接口，开发者可以确保输出张量在推理期间被正确分配和释放，从而避免潜在的内存问题并提升模型的整体性能。

### `IOutputAllocator`如何工作？

要实现 `IOutputAllocator`接口，你需要提供以下两个关键方法的实现：

- `reallocate_output_async(self, tensor_name, memory, size, alignment, stream)`：此方法负责为输出张量分配或重新分配内存。它会在推理阶段、输出张量大小已知时被调用。该方法接收张量名称、当前内存地址、新大小、对齐方式和 CUDA 流等参数，并返回新的内存地址。

- `notify_shape(self, tensor_name, shape)`：此方法用于通知分配器某个输出张量的形状发生了变化。它通常在 `reallocate_output_async()`之后被调用，以便用新的形状信息更新分配器的内部状态。

在推理过程中，TensorRT engine 会调用这些方法来管理 DDS 输出张量的内存分配。`IOutputAllocator`的实现需要确保内存分配得到妥善处理，同时考虑内存碎片、对齐和性能优化等因素。

以下是工作流程的高层概览：

1. 实例化输出分配器，并通过 [IExecutionContext.set_output_allocator()](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/ExecutionContext.html#tensorrt.IExecutionContext.set_output_allocator)将其附加到 TensorRT。

2. TensorRT engine 判定某个输出张量需要分配或重新分配内存。

3. 调用 `reallocate_output_async`为输出张量分配或重新分配内存。

4. 分配器更新其内部状态并返回新的内存地址。

5. TensorRT engine 使用新的内存地址来存储输出张量数据。

6. 调用 `notify_shape()`方法，用新的形状信息更新分配器的内部状态。

通过实现 `IOutputAllocator`接口，开发者可以创建自定义的内存分配策略，从而优化性能、减少内存碎片并提升模型推理的整体效率。

## 环境配置

建议在 TensorRT >= 10.8.0 的环境中运行这些脚本。

请按照 [TensorRT 安装指南](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html)安装 TensorRT。你需要确保 TensorRT 的 Python 绑定也已正确安装，这些绑定可通过安装 TensorRT 下载包中的 `python3-libnvinfer`和 `python3-libnvinfer-dev`软件包获得。

为了简化 TensorRT 安装，可以使用 NGC Docker 镜像，例如：

```
docker pull nvcr.io/nvidia/tensorrt:25.01-py3
```

安装 `requirements.txt`中列出的所有依赖项：

```
pip3 install -r requirements.txt
```

## 模型转换

首先，使用以下命令下载预训练的 ONNX 格式 Faster R-CNN 模型：

```
wget https://github.com/onnx/models/raw/refs/heads/main/validated/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12.onnx
```

下载 ONNX 模型后，运行以下命令为其转换为 TensorRT engine 做准备：

```
python3 modify_onnx.py \
    --input ./FasterRCNN-12.onnx \
    --output ./fasterrcnn12_trt.onnx
```

这将创建一个修改后的 ONNX 图文件，该文件已准备好转换为 TensorRT engine。

## 构建 TensorRT Engine

要构建 TensorRT engine，请运行以下命令：

```
python3 build_engine.py \
    --onnx ./fasterrcnn12_trt.onnx \
    --engine ./fasterrcnn12_trt.engine
```

## 推理

要测试构建好的 TensorRT engine，请使用以下命令下载一张测试图片：

```
wget https://onnxruntime.ai/images/demo.jpg
```

然后，使用以下命令运行推理脚本：

```
python3 infer.py \
    --engine ./fasterrcnn12_trt.engine \
    --input ./demo.jpg \
    --output ./output_dir \
    --labels labels_coco_80.txt
```

这将在测试图片上执行目标检测，并将输出保存到指定目录（本例中为 `output_dir`）。

# 变更日志

2025 年 10 月

迁移至强类型（strongly typed）API。

2025 年 8 月

移除对 < 3.10 Python 版本的支持。

2025 年 2 月

首次发布。
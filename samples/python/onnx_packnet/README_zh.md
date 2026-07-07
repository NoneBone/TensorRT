# 使用TensorRT对带自定义层的ONNX模型进行推理

 en [English](./README.md) ｜ zh_CN [简体中文](./README_zh.md)

**目录**

- [描述](#description)

- [本样例工作原理](#how-does-this-sample-work)

- [前置条件](#prerequisites)

- [运行样例](#running-the-sample)

  -   [克隆packnet仓库](#cloning-the-packnet-repository)

  -   [转换为ONNX](#conversion-to-onnx)

  -   [使用TensorRT进行推理](#inference-with-tensorrt)

  -   [样例`--help`选项](#sample-help-options)

- [额外资源](#additional-resources)

- [许可证](#license)

- [变更日志](#changelog)

- [已知问题](#known-issues)

## 描述

本样例`samplePackNet`是一个Python样例，使用TensorRT对PackNet网络执行推理。PackNet是用于自动驾驶的自监督单目深度估计网络。

## 本样例工作原理

本样例将PyTorch计算图转换为ONNX格式，并使用TensorRT内置的ONNX解析器解析ONNX计算图。本样例还演示了：

- 在ONNX计算图中使用自定义层（插件）。这些插件会通过`REGISTER_TENSORRT_PLUGIN`API自动在TensorRT中注册。

- 使用ONNX-graphsurgeon（ONNX-GS）API修改ONNX计算图中的层或子图。针对本网络，我们会转换组归一化、上采样和填充层，移除TensorRT推理时不必要的节点。

## 前置条件

1. 升级pip版本并安装样例依赖项。

   ```
   pip3 install --upgrade pip
   pip3 install -r requirements.txt
   ```

   在PowerPC系统上，你需要使用IBM的[PowerAI](https://www.ibm.com/support/knowledgecenter/SS5SF7_1.6.0/navigation/pai_install.htm)手动安装PyTorch。

## 运行样例

### 准备packnet

克隆[packnet](https://github.com/TRI-ML/packnet-sfm)仓库并更新`PYTHONPATH`。

```
git clone https://github.com/TRI-ML/packnet-sfm.git packnet-sfm
pushd packnet-sfm && git checkout tags/v0.1.2 && popd
export PYTHONPATH=$PWD/packnet-sfm # 注意：在Windows系统中，导出命令为：set PYTHONPATH=%cd%\packnet-sfm
```

### 转换为ONNX

运行以下命令将PackNet PyTorch网络转换为ONNX计算图。此步骤还包含自定义层（组归一化）的处理，以及使用ONNX-GS修改上采样和填充层。

```
python3 convert_to_onnx.py --output model.onnx
```

### 使用TensorRT进行推理

生成ONNX计算图后，使用`trtexec`工具（位于TensorRT包的`bin`目录下）对随机输入图像执行推理。

```
trtexec --onnx=model.onnx
```

更多命令行选项请参考`trtexec`工具说明。

### 样例`--help`选项

要查看所有可用选项及其说明，可使用`-h`或`--help`命令行选项。例如：

```
convert_to_onnx.py -h
```

# 额外资源

以下资源可帮助你更深入地了解PackNet网络，以及如何使用Python将模型导入TensorRT：

**PackNet**

- [3D Packing for Self-Supervised Monocular Depth Estimation](https://arxiv.org/pdf/1905.02693.pdf)

- [TRI-ML单目深度估计仓库](https://github.com/TRI-ML/packnet-sfm)

**解析器**

- [ONNX解析器](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/parsers/Onnx/pyOnnx.html)

**文档**

- [NVIDIA TensorRT样例入门](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)

- [使用Python API操作TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)

- [使用Python解析器导入模型](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_model_python)

- [NVIDIA TensorRT文档库](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# 许可证

有关使用、复制和分发的条款与条件，请参阅[TensorRT软件许可协议](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)文档。

# 变更日志

2025年8月：

- 移除对<3.10版本Python的支持。

2023年8月：

- 将ONNX版本支持更新至1.14.0

- 移除对<3.8版本Python的支持。

  2021年8月：更新样例以适配最新torch版本

  2020年6月：本样例首次发布

# 已知问题

本样例暂无已知问题
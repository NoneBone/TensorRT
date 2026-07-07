# 向您的 ONNX 网络添加自定义层实现

 en [English](./README.md) ｜ zh_CN [简体中文](./README_zh.md)

**目录**

- [描述](#description)

- [本示例的工作原理？](#how-does-this-sample-work)

- [先决条件](#prerequisites)

- [下载并预处理 ONNX 模型](#download-the-onnx-model)

- [运行示例](#running-the-sample)

- [其他资源](#additional-resources)

- [许可证](#license)

- [更新日志](#changelog)

- [已知问题](#known-issues)

## 描述

本示例 `onnx_custom_plugin`演示了如何将 C++ 编写的插件与 TensorRT Python 绑定及 ONNX 解析器结合使用。本示例使用来自 ONNX Model Zoo 的 [BiDAF Model](https://github.com/onnx/models/tree/main/text/machine_comprehension/bidirectional_attention_flow)。

## 本示例的工作原理？

本示例使用 cuBLAS 实现一个 Hardmax 层，将该实现封装为一个 TensorRT 插件（包含相应的插件创建器），并生成一个包含其代码的共享库模块。随后用户在 Python 中动态加载该库，这将导致插件注册到 TensorRT 的 PluginRegistry 中，并使其可供 ONNX 解析器使用。

本示例包含以下内容：

`plugin/`

此目录包含 Hardmax 层插件的相关文件。

`customHardmaxPlugin.cpp`

一个自定义的 TensorRT 插件实现。

`customHardmaxPlugin.h`

Hardmax 插件的头文件。

`model.py`

此脚本下载 BiDAF ONNX 模型，并使用 Onnx Graphsurgeon 替换 TensorRT 不支持的层。

`sample.py`

此脚本加载 ONNX 模型并使用 TensorRT 执行推理。

`load_plugin_lib.py`

此脚本包含一个用于在 Python 中加载 customHardmaxPlugin 库的辅助函数。

`test_custom_hardmax_plugin.py`

此脚本对照参考的 numpy 实现对 Hardmax 插件进行测试。

`requirements.txt`

此文件列出了运行此 Python 示例所需的所有 Python 包。

## 先决条件

有关具体的软件版本，请参阅 [TensorRT 安装指南](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)。

1. 安装 Python 依赖项。

```
pip3 install -r requirements.txt
```

1. [安装 CMake](https://cmake.org/download/)。

2. [安装 Cublas](https://developer.nvidia.com/cublas)。判断是否安装 `ls /usr/local/cuda/lib64 | grep cublas`

3. （针对 Windows 构建）[Visual Studio](https://visualstudio.microsoft.com/vs/older-downloads/)2017 Community 或 Enterprise 版本。

## 下载并预处理 ONNX 模型

运行模型脚本来下载 BiDAF 模型。该脚本会将 `Hardmax`层替换为一个名为 `CustomHardmax`的操作以匹配自定义插件的名称。它还会将不支持的 `Compress`节点替换为等效操作，并移除执行模型输入的字符串到整数转换的 `CategoryMapper`节点。

```
export TRT_WORKING_DIR="/root/cys/PROJECT/00-COMMON/DEMO/02-TensorRT/samples/python/onnx_custom_plugin/"
python3 model.py
```

## 运行示例

1. 构建插件及其对应的 Python 绑定。

   - 在 Linux 上，运行：

     ```
      mkdir build && pushd build
      rm -rf *
      cmake .. \
      -DTRT_INC_DIR=/root/cys/DRIVER/TensorRT-11.0.0.114/include \
      -DTRT_LIB_DIR=/root/cys/DRIVER/TensorRT-11.0.0.114/lib \
      -DNVINFER_LIB=/root/cys/DRIVER/TensorRT-11.0.0.114/lib/libnvinfer.so
      make -j$(nproc)
      popd
     ```

   - 在 Windows 上，在 Powershell 中运行以下命令，并相应替换路径：

     ```
     mkdir build; pushd build
     cmake .. -G "Visual Studio 15 Win64" `
        -DTRT_LIB=C:\path\to\tensorrt\lib `
        -DTRT_INCLUDE=C:\path\to\tensorrt\lib `
        -DCUDA_INC_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v<CUDA_VERSION>\include" `
        -DCUDA_LIB_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v<CUDA_VERSION>\lib\x64"
     # 注意：msbuild 通常位于 C:\Program Files (x86)\Microsoft Visual Studio\2017\<EDITION>\MSBuild\<VERSION>\Bin
     #   您应将此路径添加到您的 PATH 环境变量中。
     msbuild ALL_BUILD.vcxproj
     popd
     ```

   `cmake ..`命令会显示一个完整的可配置变量列表。如果某个变量被设置为 `VARIABLE_NAME-NOTFOUND`，那么您需要手动指定它或正确设置其派生变量。

2. 使用带有自定义 Hardmax 插件实现的 TensorRT 运行推理：

   ```
   python3 sample.py
   ```

3. 验证示例是否运行成功。

   ```
   === 测试中 ===
   
   输入上下文：Garry the lion is 5 years old. He lives in the savanna.
   输入查询：Where does the lion live?
   模型预测：  savanna
   
   输入上下文：A quick brown fox jumps over the lazy dog.
   输入查询：What color is the fox?
   模型预测：  brown
   ```

   该模型也可以以交互模式运行：

   ```
   python3 sample.py --interactive
   ```

   随后可以从命令行输入上下文和查询：

   ```
   === 测试中 ===
   输入上下文：Waldo wears a striped shirt. He also wears glasses.
   输入查询：Who wears glasses?
   模型预测：  waldo
   ```

# 其他资源

以下资源有助于更深入地了解如何使用 Python 入门 TensorRT：

**模型**

- [BiDAF 模型](https://allenai.github.io/bi-att-flow/)

**文档**

- [NVIDIA TensorRT 示例简介](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)

- [使用 Python API 操作 TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)

- [NVIDIA TensorRT 文档库](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# 许可证

有关使用、复制和分发的使用条款和条件，请参阅 [TensorRT 软件许可协议](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)文档。

# 更新日志

2026 年 3 月

- 将 HardmaxPlugin 从 IPluginV2DynamicExt 迁移至 IPluginV3。

2025 年 10 月

- 迁移至强类型 API。

2025 年 8 月：

- 移除对 Python 版本 < 3.10 的支持。

2024 年 1 月：

- 使用 cublasCreate 创建 cublas 句柄，不再使用 attachToContext 传入的 cublasContext 参数。

- 将 Cublas 库添加为先决条件。

2023 年 8 月：

- 将 ONNX 版本支持更新至 1.14.0。

- 移除对 Python 版本 < 3.8 的支持。

2022 年 9 月：创建并审核了本 `README.md`文件。

# 已知问题

本示例中暂无已知问题。
# Using The CuDLA API To Run A TensorRT Engine


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
   * [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
   * [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## 描述

本示例 `sampleCudla` 使用 API 构建一个仅包含单个 **ElementWise** 层的网络并生成引擎。该引擎通过 cuDLA 运行时在 **DLA 独立模式**下运行。为此，示例使用 cuDLA API 完成引擎转换、cuDLA 运行时准备以及推理执行。

## 示例工作原理

在网络构建完成后，示例会从网络数据中加载适用于 cuDLA 的模块。随后分配输入和输出张量，并在 cuDLA 中注册这些张量。当输入张量从 CPU 复制到 GPU 后，即可提交并执行 cuDLA 任务。接着等待流操作完成，并将输出缓冲区取回 CPU 以验证结果正确性。

具体步骤如下：
-   使用 TensorRT 构建单层网络。
-   调用 `cudlaCreateDevice` 创建 DLA 设备。
-   调用 `cudlaModuleLoadFromMemory` 加载供 DLA 使用的引擎内存。
-   调用 `cudaMalloc` 和 `cudlaMemRegister`，先在 GPU 上分配内存，再将 CUDA 指针注册到 DLA。
-   调用 `cudlaModuleGetAttributes` 从已加载的模块中获取模块属性。
-   调用 `cudlaSubmitTask` 提交推理任务。

### TensorRT API 层与算子

本示例使用 https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#elementwise-layer 层。更多信息，请参阅 https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers 文档。

## 前置条件

-   **平台**：本示例仅能在搭载 DLA 硬件的 aarch64 平台（Jetson 或 DRIVE）上构建和运行，不支持 x86。
-   **cuDLA 库**：`cudla` 库必须在您的 CUDA 工具包安装中可用。
-   **CMake 标志**：构建时需要包含 `-DTRT_BUILD_ENABLE_DLA=ON` 标志。

如果未启用 DLA 标志进行构建，本示例将打印以下错误信息：
```
DLA is not enabled, please compile with ENABLE_DLA=1
```
随后退出。

## 运行示例

1.  使用 CMake 并启用 DLA 标志编译示例：
	```
	cd <TensorRT root directory>
	mkdir -p build && cd build
	cmake .. -DTRT_BUILD_ENABLE_DLA=ON
	make sample_cudla
	```

	其中 `<TensorRT root directory>` 为您安装 TensorRT 的路径。

2.  运行示例以在 DLA 上执行推理。
    `./sample_cudla`

3. 验证示例是否成功运行。如果运行成功，您将看到类似以下的输出：
	```
	&&&& RUNNING TensorRT.sample_cudla # ./sample_cudla
	[I] [TRT]
	[I] [TRT] --------------- Layers running on DLA:
	[I] [TRT] [DlaLayer] {ForeignNode[(Unnamed Layer* 0) [ElementWise]]},
	[I] [TRT] --------------- Layers running on GPU:
	[I] [TRT]
	…(omit messages)
	&&&& PASSED TensorRT.sample_cudla
	```

	此输出表明示例运行成功；`PASSED`。

### 示例 `--help` 选项

要查看可用选项的完整列表及其说明，请使用 `./sample_cudla -h` 命令行选项。

## 附加资源

以下资源有助于更深入地理解 `sampleCudla`。

**文档**
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html
- https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#cudla-intro

## 许可证

有关使用、复制和分发的相关条款与条件，请参阅 https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html 文档。

## 更新日志

2022 年 6 月
这是 `README.md` 文件的首次发布。

## 已知问题

本工具暂无已知问题。
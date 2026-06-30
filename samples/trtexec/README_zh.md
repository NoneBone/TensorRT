# TensorRT 命令行封装工具：trtexec

**目录**

- [TensorRT 命令行封装工具：trtexec](#tensorrt-命令行封装工具-trtexec)
  - [简介](#简介)
  - [构建 `trtexec`](#构建-trtexec)
  - [使用 `trtexec`](#使用-trtexec)
    - [示例 1：分析自定义层性能](#示例-1分析自定义层性能)
    - [示例 2：在 DLA 上运行网络](#示例-2在-dla-上运行网络)
    - [示例 3：运行具有全维度和动态形状的 ONNX 模型](#示例-3运行具有全维度和动态形状的-onnx-模型)
    - [示例 4：收集并打印时间追踪信息](#示例-4收集并打印时间追踪信息)
    - [示例 5：通过多流调整吞吐量](#示例-5通过多流调整吞吐量)
    - [示例 6：创建强类型计划文件](#示例-6创建强类型计划文件)
  - [工具命令行参数](#工具命令行参数)
  - [其他资源](#其他资源)
- [许可证](#许可证)
- [更新日志](#更新日志)
- [已知问题](#已知问题)

## 简介

`samples`目录中包含了一个名为 `trtexec`的命令行封装工具。`trtexec`是一个无需自行开发应用程序即可快速利用 TensorRT 的工具。`trtexec`工具主要有两个用途：

- 用于对随机或用户提供的输入数据进行网络基准测试。
- 用于从模型生成序列化引擎。

**网络基准测试** - 如果您有一个保存为 ONNX 文件的模型，可以使用 `trtexec`工具测试使用 TensorRT 在网络上运行推理的性能。`trtexec`工具提供了许多选项，用于指定输入和输出、性能计时的迭代次数、允许的精度及其他选项。

**序列化引擎生成** - 如果您生成了一个保存的序列化引擎文件，可以将其引入到另一个运行推理的应用程序中。例如，您可以使用 [TensorRT Laboratory](https://github.com/NVIDIA/tensorrt-laboratory)以完全流水线的异步方式，从多个线程运行具有多个执行上下文的引擎，以测试并行推理性能。此外，在 INT8 模式下，会使用随机权重。

**使用自定义输入数据** - 默认情况下，`trtexec`将使用随机生成的输入运行推理。要为推理运行提供自定义输入，`trtexec`需要一个包含每个输入张量数据的二进制文件。建议通过 `numpy`生成此二进制文件。例如，为一个名为 `data`、形状为 `(1,3,244,244)`且类型为 `FLOAT`的 ONNX 模型创建全为 1 的自定义数据：

```
import numpy as np
data = np.ones((1,3,244,244), dtype=np.float32)
data.tofile("data.bin")
```

`trtexec`在推理期间可以使用 `--loadInputs`标志加载此二进制文件：

```sh
./trtexec --onnx=model.onnx --loadInputs=data:data.bin
```

data字段，需要根据 onnx 文件来设置。使用如下命令查询输入名称，例如查询得到首个输入 tensor 名称为 Input3，则需要写为 `--loadInputs=Input3:data.bin`

```sh
python - << 'EOF'
import onnx
m = onnx.load("/root/cys/PROJECT/00-COMMON/DEMO/02-TensorRT/data/mnist/mnist.onnx")
for i in m.graph.input:
    print("Input tensor name:", i.name)
EOF
```

输入名称可以选择用单引号括起来，以支持 Windows 上的绝对路径：

```
.\trtexec.exe --onnx=model.onnx --loadInputs='data':C:\Users\TRT\data.bin
```

## 构建 `trtexec`

`trtexec`可用于构建引擎，利用不同的 TensorRT 功能（参见命令行参数）并运行推理。`trtexec`还会测量并报告执行时间，可用于了解性能并可能定位瓶颈。

请按照 [TensorRT README](https://github.com/NVIDIA/TensorRT/)中的构建说明编译示例。

## 使用 `trtexec`

`trtexec`可以从 ONNX 格式的模型构建引擎。

### 示例 1：分析自定义层性能

您可以利用 `trtexec`分析作为 [TensorRT 插件](https://github.com/NVIDIA/TensorRT/tree/main/plugin#tensorrt-plugins)实现的自定义层。插件需要在插件注册表（`IPluginRegistry`实例）中注册才能被 TensorRT 识别。`trtexec`将加载为 TensorRT 提供插件支持的 TensorRT 标准插件库（`libnvinfer_plugin.so`/ `nvinfer_plugin.dll`）。请查看 [Non-Zero Plugins Sample](../sampleNonZeroPlugin/) 获取快速示例，或参阅 TensorRT 开发者指南的 [插件章节](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending)获取更详细的演练。

可以通过以下两种方式在 `trtexec`中使用插件：

<details>

<summary> 使用 TensorRT 随附的插件 </summary>

- 如果您使用的是 TensorRT 随附的插件（包含在 `libnvinfer_plugin.so`/ `nvinfer_plugin.dll`中），用户无需额外步骤，因为这些插件已预先注册到插件注册表中。

</details>

<details>

<summary> 使用您自己的插件 </summary>

- 如果您想定义自己的插件并让 `trtexec`将其用作网络的一部分，您应使用 TensorRT 识别的特定入口点定义自己的 *插件共享库*。然后，使用 `--dynamicPlugins`标志将共享插件库路径提供给 `trtexec`。

- 关于插件共享库及其定义方法的更多信息，请参见 [TensorRT 开发者指南](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)的 [插件共享库](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#plugin-serialization)章节。

  简而言之，有两种方法：

  1. 可以对需要静态注册的每个插件的插件创建器应用 `REGISTER_TENSORRT_PLUGIN`宏。即在插件库加载时注册。
  2. 对于动态注册，插件共享库必须暴露以下符号，这些符号将成为 TensorRT 的入口点：

  ```
  extern "C" void setLoggerFinder(ILoggerFinder* finder);
  extern "C" IPluginCreatorInterface* const* getCreators(int32_t& nbCreators)
  ```

  在上述代码中，`setLoggerFinder()`应接受一个指向 `ILoggerFinder`的指针，通过该指针可以检索 `ILogger`实例，以便在库代码内部进行日志记录。`getCreators()`应返回库包含的插件创建器数组。这些入口点的示例实现可以在 [plugin/vc/vfcCommon.cpp](./plugin/vc/vfcCommon.cpp)和 [plugin/vc/vfcCommon.h](./plugin/vc/vfcCommon.h)中找到。

  **注意**：使用 `getPluginCreators`代替 `getCreators`也是有效的，但已被弃用。

- 如果用户想先构建 TensorRT 引擎稍后运行，可以选择通过使用 `--setPluginsToSerialize`将共享插件库作为引擎的一部分进行序列化。这样做后，用户在运行构建好的引擎时无需向 `trtexec`指定 `--dynamicPlugins`。

- 有关这些标志的更多信息，请运行 `./trtexec --help`。

</details>

### 示例 2：在 DLA 上运行网络

⚠️ 警告：从 TRT11 开始，trtexec 不再支持指定 onnx 测试的精度， 通过`--fp16`, `--int8` 等参数指定，会导致报错。
- 如果需要开展量化实验，例如 int8 推理，则需要通过py源码，基于 onnx 文件(全精度)创建 int8 的 engine 文件，以便测试。
- 另外，本节的 DLA 硬件，仅嵌入式 GPU/车载 平台才拥有。

要在 NVIDIA DLA（深度学习加速器）上使用 `trtexec`以 FP16 模式运行 MNIST 网络，请执行：

```
./trtexec --onnx=data/mnist/mnist.onnx --useDLACore=1 --allowGPUFallback
```

要在 DLA 上使用 `trtexec`运行 MNIST 网络，请执行：

```
./trtexec --onnx=data/mnist/mnist.onnx --useDLACore=0 --allowGPUFallback
```

有关 DLA 的更多信息，请参见 [使用 DLA](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#dla_topic)。

### 示例 3：运行具有全维度和动态形状的 ONNX 模型

⚠️注意：ONNX 模型，可能为静态模型，此时的`--shapes`参数会导致报错。分析模型是静态（全正数）还是动态（-1）：

```sh
python - << 'EOF'
import onnx
m = onnx.load("/root/cys/PROJECT/00-COMMON/DEMO/02-TensorRT/quickstart/SemanticSegmentation/fcn-resnet101.onnx")
for i in m.graph.input:
    print(i.name, [d.dim_value if d.dim_value > 0 else -1 for d in i.type.tensor_type.shape.dim])
EOF
```

在全维度模式下运行 ONNX 模型（静态输入形状）：

```
./trtexec --onnx=model.onnx
```

以下示例假设有一个动态输入名称为 `input`且维度为 `[-1, 3, 244, 244]`的 ONNX 模型。

在全维度模式下使用给定输入形状运行 ONNX 模型：

```
./trtexec --onnx=model.onnx --shapes=input:32x3x244x244
```

使用一系列可能的输入形状对 ONNX 模型进行基准测试：

```
./trtexec --onnx=model.onnx --minShapes=input:1x3x244x244 --optShapes=input:16x3x244x244 --maxShapes=input:32x3x244x244 --shapes=input:5x3x244x244
```

### 示例 4：收集并打印时间追踪信息

运行时，`trtexec`会打印测量的性能，但也可以将测量结果导出到 json 文件：

```
./trtexec --onnx=data/mnist/mnist.onnx --exportTimes=trace.json
```

一旦追踪信息存储在文件中，就可以使用 `tracer.py`工具进行打印。该工具以不同形式打印输入、计算和输出的时间戳及持续时间：

```
./tracer.py trace.json
```

同样，也可以打印性能分析结果并将其存储在 json 文件中。可以使用 `profiler.py`工具读取并打印 json 文件中的性能分析结果。

### 示例 5：通过多流调整吞吐量

Tuning throughput 可能需要运行多个并发的执行流。例如，当达到的延迟远低于所需阈值时，即使牺牲一些延迟，我们也可以提高吞吐量。例如，保存具有不同精度的引擎，并假设两者都在 2 毫秒内执行（延迟阈值为 2 毫秒）：

```
trtexec --onnx=resnet50.onnx --saveEngine=g1.trt --int8 --skipInference
trtexec --onnx=resnet50.onnx --saveEngine=g2.trt --best --skipInference
```

现在，可以尝试使用保存的引擎，找到低于 2 毫秒且能最大化吞吐量的精度/流组合：

```
trtexec --loadEngine=g1.trt --streams=1 # 983qps 1.00ms
trtexec --loadEngine=g1.trt --streams=2 # 1324qps 1.52ms
trtexec --loadEngine=g1.trt --streams=3 # 1373qps 2.13ms
trtexec --loadEngine=g1.trt --streams=4 # 1432qps 2.74ms
trtexec --loadEngine=g2.trt --streams=2 # 
```

### 示例 6：创建强类型计划文件

此标志将创建一个带有 `NetworkDefinitionCreationFlag::kSTRONGLY_TYPED`标志的网络，其中张量数据类型是从网络输入类型和算子类型规范推断出来的。使用此选项时不允许使用特定的构建器精度标志（如 `--int8`或 `--best`）。

```
./trtexec --onnx=model.onnx --stronglyTyped
```

## 工具命令行参数

要查看可用选项的完整列表及其说明，请执行 `./trtexec --help`命令。

**注意：** 指定 `--safe`参数会将安全模式开关切换为 `ON`。默认情况下，未指定 `--safe`参数；安全模式开关为 `OFF`。如果开关设置为 `ON`，则 `--safe`子集中包含的层和参数将受到限制。在 TensorRT 安全运行时可用之前，该开关用于原型设计安全限制流程。在使用标准 TensorRT 包加载或保存安全引擎时需要此参数。有关更多信息，请参见 TensorRT 开发者指南中的 [使用汽车安全章节](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#working_auto_safety)。

## 其他资源

以下资源提供了关于 `trtexec`的更多详细信息：

**文档**

- [NVIDIA trtexec](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#trtexec)
- [TensorRT 示例支持指南](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html)
- [NVIDIA TensorRT 文档库](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# 许可证

有关使用、复制和分发的条款和条件，请参阅 [TensorRT 软件许可协议](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)文档。

# 更新日志

2019 年 4 月

这是此 `README.md`文件的首次发布。

# 已知问题

本示例中暂无已知问题。
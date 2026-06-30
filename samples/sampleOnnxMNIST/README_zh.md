# “Hello World”：从 ONNX 开始使用 TensorRT

**目录**
- #description
- #how-does-this-sample-work
	* #converting-the-onnx-model-to-a-tensorrt-network
	* #building-the-engine
	* #running-inference
	* #tensorrt-api-layers-and-ops
- #running-the-sample
	* #sample-help-options
- #additional-resources
- #license
- #changelog
- #known-issues

## 描述

本示例 `sampleOnnxMNIST` 将一个在 MNIST 数据集上训练、并以开放神经网络交换（ONNX）格式保存的模型转换为 TensorRT 网络，并在该网络上运行推理。

ONNX 是一种用于表示深度学习模型的标准，它使得模型能够在不同框架之间迁移。

## 示例工作原理

本示例从 MNIST 网络的 ONNX 模型创建并运行 TensorRT 引擎。它演示了 TensorRT 如何将 ONNX 模型作为输入来创建网络。

具体来说，本示例：
- #converting-the-onnx-model-to-a-tensorrt-network
- #building-the-engine
- #running-inference

### 将 ONNX 模型转换为 TensorRT 网络

可以使用 ONNX 解析器将模型文件转换为 TensorRT 网络。解析器可以使用网络定义和日志记录器对象进行初始化。

`auto parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());`

然后将 ONNX 模型文件连同日志级别一起传递给解析器：

```
if (!parser->parseFromFile(model_file, static_cast<int>(sample::gLogger.getReportableSeverity())))
{
	  string msg("failed to parse onnx file");
	  sample::gLogger->log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
	  exit(EXIT_FAILURE);
}
```

通过解析模型构建出 TensorRT 网络后，就可以构建 TensorRT 引擎来运行推理了。

### 构建引擎

要构建引擎，请创建构建器并传入为 TensorRT 创建的日志记录器，该记录器用于报告网络中的错误、警告和信息性消息：
`IBuilder* builder = createInferBuilder(sample::gLogger);`

要从生成的 TensorRT 网络构建引擎，请发出以下调用：
`std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};`

构建引擎后，请验证引擎是否正常运行，确认输出是否符合预期。本示例的输出格式应与 `sampleMNIST` 的输出相同。

### 运行推理

要使用创建的引擎运行推理，请参阅 https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_c。

**注意：** 预处理数据并将其转换为网络接受的格式非常重要。在本示例中，输入样本为 PGM（便携式灰度图）格式。模型期望的输入图像尺寸为 `1x28x28`，且像素值缩放至 `[0,1]` 之间。

### TensorRT API 层与操作

本示例中使用了以下层。有关这些层的更多信息，请参阅 https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers 文档。

https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer
激活层实现逐元素的激活函数。具体而言，本示例使用类型为 `kRELU` 的激活层。

https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#convolution-layer
卷积层计算二维（通道、高度和宽度）卷积，可带或不带偏置。

https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#matrixmultiply-layer
矩阵乘法层实现矩阵乘法运算。
（https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#fullyconnected-layer 自 8.4 版本起已弃用。
全连接层的偏置可以通过 `SUM` 操作的 https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#elementwise-layer 添加。）

https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#pooling-layer
池化层在通道内实现池化。支持的池化类型包括 `最大池化`、`平均池化` 和 `最大-平均混合池化`。

https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#scale-layer
缩放层实现基于常量值的逐张量、逐通道或逐元素的仿射变换和/或指数运算。

https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#shuffle-layer
洗牌层实现张量的重塑和转置操作。

## 先决条件
1. 准备示例数据

请参阅主示例 README 中的 ../README.md#preparing-sample-data。

## 运行示例

1. 按照 https://github.com/NVIDIA/TensorRT/ 中的构建说明编译示例。

2.  运行示例以从 ONNX 模型构建并运行 MNIST 引擎。
	```
	./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]
	```

3.  验证示例是否成功运行。如果示例成功运行，您应该会看到类似于以下的输出：
	```
	&&&& RUNNING TensorRT.sample_onnx_mnist # ./sample_onnx_mnist
	----------------------------------------------------------------
	Input filename: ../../../../../../data/samples/mnist/mnist.onnx
	ONNX IR version: 0.0.3
	Opset version: 1
	Producer name: CNTK
	Producer version: 2.4
	Domain:
	Model version: 1
	Doc string:
	----------------------------------------------------------------
	[I] Input:
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@*.  .*@@@@@@@@@@@
	@@@@@@@@@@*.     +@@@@@@@@@@
	@@@@@@@@@@. :#+   %@@@@@@@@@
	@@@@@@@@@@.:@@@+  +@@@@@@@@@
	@@@@@@@@@@.:@@@@:  +@@@@@@@@
	@@@@@@@@@@=%@@@@:  +@@@@@@@@
	@@@@@@@@@@@@@@@@#  +@@@@@@@@
	@@@@@@@@@@@@@@@@*  +@@@@@@@@
	@@@@@@@@@@@@@@@@:  +@@@@@@@@
	@@@@@@@@@@@@@@@@:  +@@@@@@@@
	@@@@@@@@@@@@@@@*  .@@@@@@@@@
	@@@@@@@@@@%**%@.  *@@@@@@@@@
	@@@@@@@@%+.  .:  .@@@@@@@@@@
	@@@@@@@@=  ..    :@@@@@@@@@@
	@@@@@@@@:  *@@:  :@@@@@@@@@@
	@@@@@@@%   %@*    *@@@@@@@@@
	@@@@@@@%   ++ ++  .%@@@@@@@@
	@@@@@@@@-    +@@-  +@@@@@@@@
	@@@@@@@@=  :*@@@#  .%@@@@@@@
	@@@@@@@@@+*@@@@@%.   %@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	[I] Output:
	Prob 0 0.0000 Class 0:
	Prob 1 0.0000 Class 1:
	Prob 2 1.0000 Class 2: **********
	Prob 3 0.0000 Class 3:
	Prob 4 0.0000 Class 4:
	Prob 5 0.0000 Class 5:
	Prob 6 0.0000 Class 6:
	Prob 7 0.0000 Class 7:
	Prob 8 0.0000 Class 8:
	Prob 9 0.0000 Class 9:

	&&&& PASSED TensorRT.sample_onnx_mnist # ./sample_onnx_mnist
	```

	此输出表明示例已成功运行；显示 PASSED。

### 示例 `--help` 选项

要查看可用选项的完整列表及其描述，请使用 `-h` 或 `--help` 命令行选项。

# 附加资源

以下资源有助于更深入地理解 ONNX 项目和 MNIST 模型：

**ONNX**
- https://github.com/onnx/onnx
- https://github.com/onnx/onnx-tensorrt

**模型**
- https://github.com/onnx/models/tree/main/validated/vision/classification/mnist
- https://github.com/onnx/models

**文档**
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html

# 许可证

有关使用、复制和分发的条款和条件，请参阅 https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html 文档。

# 更新日志

2025年10月
迁移至强类型 API。

2019年3月
重新创建、更新并审核了此 `README.md` 文件。

# 已知问题

本示例中无已知问题。
# TensorRT 中的动态形状手写数字识别

**目录**
- #描述
- #本示例如何工作
    * #创建预处理网络
    * #解析-onnx-mnist-模型
    * #构建引擎
    * #运行推理
	* #tensorrt-api-层与算子
- #准备示例数据
- #运行示例
	* #示例---help-选项
- #其他资源
- #许可证
- #更新日志
- #已知问题

## 描述

本示例 `sampleDynamicReshape` 演示了如何在 TensorRT 中使用动态输入维度。它创建一个接收动态形状输入并将其调整为 ONNX MNIST 模型所需固定尺寸输入的引擎。更多信息，请参阅 TensorRT 开发者指南中的 https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#work_dynamic_shapes。

## 本示例如何工作？

本示例创建一个引擎，用于将动态维度的输入调整为 ONNX MNIST 模型可接受的尺寸。

具体而言，本示例：
-   创建一个具有动态输入维度的网络，作为模型的预处理器
-   解析 ONNX MNIST 模型以创建第二个网络
-   为两个网络构建引擎
-   使用两个引擎运行推理

### 创建预处理网络

首先，创建一个支持全维度的网络：
`auto preprocessorNetwork = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));`

接下来，添加一个接受动态形状输入的输入层，后接一个调整输入形状至模型期望尺寸的缩放层：
```
auto input = preprocessorNetwork->addInput("input", nvinfer1::DataType::kFLOAT, Dims4{-1, 1, -1, -1});
auto resizeLayer = preprocessorNetwork->addResize(*input);
resizeLayer->setOutputDimensions(mPredictionInputDims);
preprocessorNetwork->markOutput(*resizeLayer->getOutput(0));
```

`-1` 维度表示将在运行时提供的维度。

### 解析 ONNX MNIST 模型

首先，创建一个空的、支持全维度的网络及解析器：
```
auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
auto parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());
```

接着，解析模型文件以填充网络：
```
parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
```

### 构建引擎

在构建预处理器引擎时，还需提供一个优化配置，以便 TensorRT 知道针对哪些输入形状进行优化：
```
auto preprocessorConfig = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
auto profile = builder->createOptimizationProfile();
```

`OptProfileSelector::kOPT` 指定优化配置的目标维度，而 `OptProfileSelector::kMIN` 和 `OptProfileSelector::kMAX` 指定配置有效的维度最小值和最大值：
```
profile->setDimensions(input->getName(), OptProfileSelector::kMIN, Dims4{1, 1, 1, 1});
profile->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims4{1, 1, 28, 28});
profile->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims4{1, 1, 56, 56});
preprocessorConfig->addOptimizationProfile(profile);
```

使用配置运行引擎构建：
```cpp
auto preprocessorPlan = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*preprocessorNetwork, *preprocessorConfig));
if (!preprocessorPlan)
{
    sample::gLogError << "预处理器序列化引擎构建失败。" << std::endl;
    return false;
}

mPreprocessorEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
    runtime->deserializeCudaEngine(preprocessorPlan->data(), preprocessorPlan->size()));
if (!mPreprocessorEngine)
{
    sample::gLogError << "预处理器引擎反序列化失败。" << std::endl;
    return false;
}
```

对于 MNIST 模型，在网路末端附加一个 Softmax 层，将 Softmax 轴设为 1（因为在全维度模式下网络输出形状为 [1, 10]），并用 Softmax 替换原有的网络输出：
```cpp
auto softmax = network->addSoftMax(*network->getOutput(0));
softmax->setAxes(1 << 1);
network->unmarkOutput(*network->getOutput(0));
network->markOutput(*softmax->getOutput(0));
```

最后，正常构建：
```
auto predictionPlan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
if (!predictionPlan)
{
    sample::gLogError << "预测序列化引擎构建失败。" << std::endl;
    return false;
}

mPredictionEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
    runtime->deserializeCudaEngine(predictionPlan->data(), predictionPlan->size()));
if (!mPredictionEngine)
{
    sample::gLogError << "预测引擎反序列化失败。" << std::endl;
    return false;
}
```

### 运行推理

推理期间，首先将输入缓冲区复制到设备：
```
CHECK(cudaMemcpy(mInput.deviceBuffer.data(), mInput.hostBuffer.data(), mInput.hostBuffer.nbBytes(), cudaMemcpyHostToDevice));
```

由于预处理器引擎接受动态形状，需向执行上下文指定当前输入的实际形状：
`mPreprocessorContext->setInputShape(inputTensorName, inputDims);`，其中 `inputTensorName` 是绑定索引 0 上的输入张量名称。

接下来，使用 `executeV2` 函数运行预处理器。示例将预处理器引擎的输出直接写入 MNIST 引擎的输入设备缓冲区：
```
std::vector<void*> preprocessorBindings = {mInput.deviceBuffer.data(), mPredictionInput.data()};
bool status = mPreprocessorContext->executeV2(preprocessorBindings.data());
```

然后，运行 MNIST 引擎：
```
std::vector<void*> predicitonBindings = {mPredictionInput.data(), mOutput.deviceBuffer.data()};
status = mPredictionContext->executeV2(predicitonBindings.data());
```

最后，将输出复制回主机：
```
CHECK(cudaMemcpy(mOutput.hostBuffer.data(), mOutput.deviceBuffer.data(), mOutput.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost));
```

### TensorRT API 层与算子

本示例使用以下层。有关这些层的更多信息，请参阅 https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers 文档。

https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#resize-layer
`IResizeLayer` 实现对输入张量的缩放操作。

## 先决条件
1. 参见主示例 README 中的 ../README.md#preparing-sample-data。

## 运行示例

1. 按照 https://github.com/NVIDIA/TensorRT/ 中的构建说明编译示例。

2.  运行示例。
    ```bash
    ./sample_dynamic_reshape [-h 或 --help] [-d 或 --datadir=<数据目录路径>] [--useDLACore=<整数>]
    ```

    例如：
    ```bash
    ./sample_dynamic_reshape --datadir $TRT_DATADIR/mnist
    ```

3. 验证示例是否成功运行。如果示例运行成功，您应看到类似于以下的输出：
    ```
  	&&&& 运行中 TensorRT.sample_dynamic_reshape # ./sample_dynamic_reshape
    ----------------------------------------------------------------
    输入文件名：   ../../../../../data/samples/mnist/mnist.onnx
    ONNX IR 版本：  0.0.3
    Opset 版本：    8
    生产者名称：    CNTK
    生产者版本： 2.5.1
    域：           ai.cntk
    模型版本：    1
    文档字符串：  
    ----------------------------------------------------------------
    [W] [TRT] onnx2trt_utils.cpp:214: 您的 ONNX 模型包含 INT64 权重生成，而 TensorRT 原生不支持 INT64。尝试向下转换为 INT32。
    [W] [TRT] onnx2trt_utils.cpp:214: 您的 ONNX 模型包含 INT64 权重生成，而 TensorRT 原生不支持 INT64。尝试向下转换为 INT32。
    [I] [TRT] 检测到 1 个输入和 1 个输出网络张量。
    [I] [TRT] 检测到 1 个输入和 1 个输出网络张量。
    [I] 预处理器引擎中的轮廓维度：
    [I]     最小值 = (1, 1, 1, 1)
    [I]     最优值 = (1, 1, 28, 28)
    [I]     最大值 = (1, 1, 56, 56)
    [I] 输入：
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@*.  .*@@@@@@@@@@@
    @@@@@@@@@@*.     +@@@@@@@@@@
    @@@@@@@@@@. :#+   %@@@@@@@@@
    @@@@@@@@@@.:@@@+  +@@@@@@@@@
    @@@@@@@@@@.:@@@@: +@@@@@@@@@
    @@@@@@@@@@=%@@@@: +@@@@@@@@@
    @@@@@@@@@@@@@@@@# +@@@@@@@@@
    @@@@@@@@@@@@@@@@* +@@@@@@@@@@
    @@@@@@@@@@@@@@@@: +@@@@@@@@@
    @@@@@@@@@@@@@@@@: +@@@@@@@@@
    @@@@@@@@@@@@@@@* .@@@@@@@@@@
    @@@@@@@@@@%**%@. *@@@@@@@@@@
    @@@@@@@@%+.  .: .@@@@@@@@@@@
    @@@@@@@@=  ..   :@@@@@@@@@@@
    @@@@@@@@: *@@:  :@@@@@@@@@@@
    @@@@@@@%  %@*    *@@@@@@@@@@
    @@@@@@@%  ++  ++ .%@@@@@@@@@
    @@@@@@@@-    +@@- +@@@@@@@@@
    @@@@@@@@=  :*@@@# .%@@@@@@@@
    @@@@@@@@@+*@@@@@%.  %@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@

    [I] 输出：
    [I]  概率 0  0.0000 类别 0: 
    [I]  概率 1  0.0000 类别 1: 
    [I]  概率 2  1.0000 类别 2: **********
    [I]  概率 3  0.0000 类别 3: 
    [I]  概率 4  0.0000 类别 4: 
    [I]  概率 5  0.0000 类别 5: 
    [I]  概率 6  0.0000 类别 6: 
    [I]  概率 7  0.0000 类别 7: 
    [I]  概率 8  0.0000 类别 8: 
    [I]  概率 9  0.0000 类别 9: 
    &&&& 通过 TensorRT.sample_dynamic_reshape # ./sample_dynamic_reshape
    ```

    此输出表明示例运行成功；`通过`。

### 示例 `--help` 选项

要查看可用选项的完整列表及其描述，请使用 `-h` 或 `--help` 命令行选项。

# 其他资源

以下资源有助于更深入地理解动态形状。

**ONNX**
- https://github.com/onnx/onnx
- https://github.com/onnx/onnx-tensorrt

**模型**
- https://github.com/onnx/models/tree/main/validated/vision/classification/mnist
- https://github.com/onnx/models

**文档**
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html

# 许可证

有关使用、复制和分发的条款和条件，请参阅 https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html 文档。

# 更新日志

2025 年 10 月
迁移至强类型 API。

2020 年 2 月
这是 `README.md` 文件和示例的第二次发布。

# 已知问题

本示例中暂无已知问题。
# 框架比较

## 介绍

你可以使用 `run` 用于比较不同框架下模型的子工具。
最简单的情况下，您可以提供一个模型，以及一个或多个框架标志。
默认情况下，它将生成合成输入数据，并使用该数据运行推理。
指定框架，然后比较指定框架的输出。

## 运行示例

在这个例子中，我们将概述各种常见的用例。 `run` 子工具：

- [Comparing TensorRT And ONNX-Runtime Outputs](#comparing-tensorrt-and-onnx-runtime-outputs)
- [Comparing TensorRT Precisions](#comparing-tensorrt-precisions)
- [Changing Tolerances](#changing-tolerances)
- [Changing Comparison Metrics](#changing-comparison-metrics)
- [Comparing Per-Layer Outputs Between ONNX-Runtime And TensorRT](#comparing-per-layer-outputs-between-onnx-runtime-and-tensorrt)

### 比较 TensorRT 和 ONNX-Runtime 的输出

在 Polygraphy 中使用这两个框架运行模型并执行输出
比较：

```bash
polygraphy run dynamic_identity.onnx --trt --onnxrt
```

这 `dynamic_identity.onnx` 模型具有动态输入形状。默认情况下，
多导成像技术将覆盖模型中任何动态输入尺寸。
`constants.DEFAULT_SHAPE_VALUE` （定义为 `1`）并警告你：

<!-- Polygraphy Test: Ignore Start -->
```
[W]     Input tensor: X (dtype=DataType.FLOAT, shape=(1, 2, -1, -1)) | No shapes provided; Will use shape: [1, 2, 1, 1] for min/opt/max in profile.
[W]     This will cause the tensor to have a static shape. If this is incorrect, please set the range of shapes for this input tensor.
```
<!-- Polygraphy Test: Ignore End -->

为了抑制此消息并明确提供输入形状
测谎，使用 `--input-shapes` 选项：

```
polygraphy run dynamic_identity.onnx --trt --onnxrt \
    --input-shapes X:[1,2,4,4]
```

### 比较TensorRT精度

构建一个具有降低精度层的 TensorRT 引擎，以便与……进行比较。
ONNXRT，请使用受支持的精度标志之一（例如） `--tf32`， `--fp16`，`--int8`， ETC。）。
例如：

```bash
polygraphy run dynamic_identity.onnx --trt --fp16 --onnxrt \
    --input-shapes X:[1,2,4,4]
```

> 警告：要达到可接受的 INT8 精度，通常需要额外的校准步骤：
  参见 [developer guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)
  以及相关说明 [how to do calibration](../../../../examples/cli/convert/01_int8_calibration_in_tensorrt)
  使用命令行进行测谎。

### 改变公差

默认容差由 `run` 通常适用于FP32精度
但可能不适用于精度要求较低的情况。为了放宽公差，
你可以使用 `--atol` 和 `--rtol` 可设置绝对值和相对值
分别为容差。

### 改变比较指标

你可以使用 `--check-error-stat` 可选择更改所用指标
比较。默认情况下，Polygraphy 使用“逐元素”度量。
（`--check-error-stat elemwise`）。

其他可能的指标 `--check-error-stat` 是 `mean`， `median`， 和 `max`， 哪个
分别比较张量的平均值、中位数和最大绝对/相对误差。

为了更好地理解这一点，假设我们是
比较两个输出 `out0` 和 `out1`测谎需要
这些张量的逐元素绝对差值和相对差值：

<!-- Polygraphy Test: Ignore Start -->
```
absdiff = out0 - out1
reldiff = absdiff / abs(out1)
```
<!-- Polygraphy Test: Ignore End -->

然后，对于每个索引 `i` 在输出结果中，测谎仪会检查是否
`absdiff[i] > atol and reldiff[i] > rtol`如果任何指数满足此条件，
那么比较就会失败。这比比较最大值要宽松得多。
整个张量的绝对误差和相对误差（`--check-error-stat max`因为如果
*不同的*索引 `i` 和 `j` 满足 `absdiff[i] > atol` 和 `reldiff[j] > rtol`，
然后 `max` 比较将会失败，但是 `elemwise` 比较可能
经过。

综合以上所有内容，以下示例运行起来…… `median` 比较
使用 FP16 和 ONNX-Runtime 的 TensorRT，采用绝对和相对容差 `0.001`：

```bash
polygraphy run dynamic_identity.onnx --trt --fp16 --onnxrt \
    --input-shapes X:[1,2,4,4] \
    --atol 0.001 --rtol 0.001 --check-error-stat median
```

> 您还可以为每个输出指定值。 `--atol`/`--rtol`/`--check-error-stat`。
  请查看帮助输出。 `run` 子工具提供更多信息。

### 比较 ONNX-Runtime 和 TensorRT 的逐层输出

当网络输出不匹配时，比较每一层的输出可能很有用。
要找出错误所在，可以使用以下方法： `--trt-outputs`
和 `--onnx-outputs` 分别对应不同的选项。这些选项接受一个或多个选项。
输出名称作为参数。特殊值 `mark all` 表明所有
应该对模型中的张量进行比较：

```bash
 polygraphy run dynamic_identity.onnx --trt --onnxrt \
     --trt-outputs mark all \
     --onnx-outputs mark all
```

为了更轻松地找到第一个不匹配的输出，您可以使用 `--fail-fast`
此选项将导致工具在第一次不匹配后退出
输出。

请注意使用 `--trt-outputs mark all` 有时会扰乱生成的
由于时序、图层融合选择和格式的差异，引擎会有所不同。
约束条件可能会掩盖故障。在这种情况下，您可能需要使用……
更复杂的方法来分析失效模型并生成一个简化的模型
可重现错误的测试用例。请参阅[减少 ONNX 故障]
有关模型的教程，请参阅[../../../../examples/cli/debug/02_reducing_failing_onnx_models](../../../../examples/cli/debug/02_reducing_failing_onnx_models)。
如何利用测谎技术做到这一点。

## 延伸阅读

* 在某些情况下，您可能需要对多次测谎结果进行比较。
  （例如，在比较预构建的 TensorRT 引擎的输出时）
  [Polygraphy network script](../../../../examples/cli/run/04_defining_a_tensorrt_network_or_config_manually)
  （针对 ONNX-Runtime）。参见 [Comparing Across Runs](../../../../examples/cli/run/02_comparing_across_runs) 有关如何操作的教程
  完成这项任务。

* 有关在 TensorRT 中使用动态形状的更多详细信息：
  * 看 [Dynamic Shapes in TensorRT](../../../../examples/cli/convert/03_dynamic_shapes_in_tensorrt/) 如何指定
    用于通过 Polygraphy CLI 与引擎配合使用的优化配置文件
  * 看 [TensorRT and Dynamic Shapes](../../../../examples/api/07_tensorrt_and_dynamic_shapes/) 详情请见
    如何使用 Polygraphy API 实现这一点

* 有关如何提供实际输入数据的详细信息，请参阅 [Comparing with Custom Input Data](../05_comparing_with_custom_input_data/)。

* 看 [Debugging TensorRT Accuracy Issues](../../../../how-to/debug_accuracy.md) 有关如何使用多导成像技术调试精度故障的更详细教程。

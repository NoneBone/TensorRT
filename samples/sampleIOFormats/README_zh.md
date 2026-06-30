# Specifying I/O Formats


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample-help-options)
- [Preparing sample data](#preparing-sample-data)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## 描述

本示例 `sampleIOFormats` 使用了在 https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md 上训练的 ONNX 模型，并利用 TensorRT 执行引擎构建和推理。随后会将输出的正确性与黄金参考值进行比较。具体来说，它展示了如何使用 API 显式地将输入格式指定为 `TensorFormat::kLINEAR`、`TensorFormat::kHWC` 和 `TensorFormat::kCHW32`（针对 Float32）。

## 本示例的工作原理？

通过调用 `ITensor::setAllowedFormats` 来指定预期支持的格式。

	```
	bool SampleIOFormats::build(int dataWidth)
	{
		...

		network->getInput(0)->setAllowedFormats(static_cast<TensorFormats>(1 << static_cast<int>(mTensorFormat)));
		...
	}
	```

## 先决条件
1. 准备示例数据
请参阅主示例 README 中的 ../README.md#preparing-sample-data。

## 运行示例

1. 按照 https://github.com/NVIDIA/TensorRT/ 中的构建说明编译示例。

2.  对从 0 到 9 的数字循环运行推理：
    ```bash
    ./sample_io_formats --datadir=<path/to/data> --useDLACore=N
    ```

    例如：
    ```bash
    ./sample_io_formats --datadir $TRT_DATADIR/mnist
    ```

3.  验证所有 10 个数字均正确匹配。如果示例运行成功，您将看到类似于以下的输出：
	```
	&&&& RUNNING TensorRT.sample_io_formats # ./sample_io_formats
	[I] Build TRT engine with different IO data type and formats. Ensure that built engine abide by them
	[I] Testing datatype FP32 with format kLINEAR
	[I] Building and running a GPU inference engine with specified I/O formats.
	... (omitted message)
	[I] Testing datatype FP32 with format kHWC
	[I] Building and running a GPU inference engine with specified I/O formats.
	... (omitted message)
	[I] Testing datatype FP32 with format kCHW32
	[I] Building and running a GPU inference engine with specified I/O formats.
	... (omitted message)
	&&&& PASSED TensorRT.sample_io_formats
	```
	此输出表明示例运行成功；显示 `PASSED`。

### 示例 `--help` 选项

要查看可用选项的完整列表及其描述，请使用 `-h` 或 `--help` 命令行选项。

## 其他资源

以下资源有助于更深入地理解本示例：

**模型**
- https://keras.io/datasets/#mnist-database-of-handwritten-digits

**文档**
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html

## 许可证

有关使用、复制和分发的使用条款和条件，请参阅 https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html 文档。

## 更新日志

**2025 年 10 月**
- 迁移至强类型 API。

**2022 年 8 月**
- 将代码从解析 `caffe` 模型迁移至 `onnx` 模型。

**2021 年 10 月**
- 将名称和主题从“无重格式化（reformat-free）”更改为“I/O 格式”，因为 `BuilderFlag::kSTRICT_TYPES` 已被弃用。“无重格式化 I/O”（参见 `BuilderFlag::kDIRECT_IO`）通常是适得其反且脆弱的，因为它限制了优化器选择最快实现的能力，并且依赖于特定目标上可用的内核。

**2019 年 6 月**
- 这是 `README.md` 文件和示例的首次发布。

## 已知问题

本示例中暂无已知问题。
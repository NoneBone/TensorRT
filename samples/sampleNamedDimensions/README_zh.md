# Working with ONNX models with named input dimensions


**Table Of Contents**
- [Description](#description)
- [Running the sample](#running-the-sample)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## 描述

本示例 `sampleNamedDimensions` 演示了如何在 TensorRT 中处理带有命名输入维度的 ONNX 模型。

ONNX 具有命名维度参数的概念：两个具有相同命名维度参数的网络输入被视为相等。TensorRT 支持此功能，它会检查优化配置文件中的这些维度是否具有重叠区间，并在运行时确保它们具有相同的值。

在这里，我们合成创建了一个 ONNX 模型，该模型由一个 https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat 层组成，该层接收两个二维输入张量：
```
input0      input1
    \         /
     \       /
      --------
      |Concat|
      --------
          |
          |
       output
```
拼接操作在第零个轴上执行，因此只需要输入张量的第一个维度相同。但是，由于两个输入的维度均为 `[n_rows, 8]`，命名维度 `n_rows` 还要求两个输入张量的第零个维度也必须匹配。

## 运行示例

1.  按照 https://github.com/NVIDIA/TensorRT 构建 TensorRT OSS 时，本示例会被一并编译。名为 `sample_named_dimensions` 的二进制文件将在输出目录中生成。

2.  运行以下命令生成 ONNX 模型文件：
	```
	python3 create_model.py
	```
	这将创建一个名为 `concat_layer.onnx` 的文件。

3. 运行示例以从 ONNX 模型构建并运行引擎。
	```
	./sample_named_dimensions [-h 或 --help] [-d 或 --datadir=<数据目录路径>]
	```

3.  验证示例是否成功运行。如果成功，您应该看到类似于以下的输出：
	```
	&&&& RUNNING TensorRT.sample_named_dimensions [TensorRT v8500] # build/x86_64-gnu/sample_named_dimensions
	[I] [TRT] ----------------------------------------------------------------
	[I] [TRT] Input filename:   ../trt/samples/sampleNamedDimensions/concat_layer.onnx
	[I] [TRT] ONNX IR version:  0.0.7
	[I] [TRT] Opset version:    11
	[I] [TRT] Producer name:
	[I] [TRT] Producer version:
	[I] [TRT] Domain:
	[I] [TRT] Model version:    0
	[I] [TRT] Doc string:
	[I] [TRT] ----------------------------------------------------------------
	[I] Input0:
	-4.17896 4.21201 -8.6982 9.33153 -4.90741 1.1953 9.45208 1.04329
	-5.47509 0.150872 -4.29573 1.72331 3.69642 5.73303 -4.89766 5.00559
	
	[I] Input1:
	9.01907 3.57581 -1.36986 -3.22044 -5.90874 -8.11433 2.38472 -0.0868187
	0.842402 -1.75138 4.55962 -6.38946 -7.73614 -1.26044 -4.23012 4.33806
	
	[I] Output:
	-4.17896 4.21201 -8.6982 9.33153 -4.90741 1.1953 9.45208 1.04329
	-5.47509 0.150872 -4.29573 1.72331 3.69642 5.73303 -4.89766 5.00559
	9.01907 3.57581 -1.36986 -3.22044 -5.90874 -8.11433 2.38472 -0.0868187
	0.842402 -1.75138 4.55962 -6.38946 -7.73614 -1.26044 -4.23012 4.33806
	
	&&&& PASSED TensorRT.sample_named_dimensions [TensorRT v8500] # build/x86_64-gnu/sample_named_dimensions
	```

### 示例 `--help` 选项

要查看可用选项的完整列表及其描述，请使用 `-h` 或 `--help` 命令行选项。

# 附加资源

以下资源有助于更深入地了解 ONNX 项目中的命名输入维度功能：

**ONNX**
- https://github.com/onnx/onnx
- https://github.com/onnx/onnx-tensorrt

**文档**
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html

# 许可证

有关使用、复制和分发的条款和条件，请参阅 https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html 文档。

# 更新日志

2025 年 10 月
迁移至强类型 API。

2022 年 6 月
重新创建、更新并审查了此 `README.md` 文件。

# 已知问题

本示例中暂无已知问题。
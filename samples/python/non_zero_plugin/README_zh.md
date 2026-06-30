# 基于 Python 的 TensorRT NonZero 插件（使用 IPluginV3）

## 描述

本样例 `non_zero_plugin`实现了一个基于 Python 的 NonZero 操作插件，可配置使用 `CUDA Python`或 `PyTorch`后端。

NonZero 操作用于查找输入张量中非零元素的索引。

## 本样例如何工作？

本样例创建并运行一个 TensorRT engine，该 engine 由一个包含单个 NonZeroPlugin 节点的网络构建而成。它演示了如何使用 Python 实现具有**数据依赖输出形状**的自定义层并将其添加到 TensorRT 网络中。

### 使用 IPluginV3 接口实现 NonZero 插件

在 `IPluginV3`（及相关接口）出现之前，TensorRT 插件的输出形状无法依赖于输入数据的值（只能依赖于输入形状）。暴露了 `IPluginV3`构建能力的 `IPluginV3OneBuild`为此类数据依赖的输出形状提供了支持。

本样例中的 `NonZeroPlugin`用于处理形状为 R×C的二维输入张量。假设该张量包含 K个非零元素，并且需要按行顺序（每组索引位于一行）获取这些非零索引，则输出形状将为 K×2。

输出形状通过 `IPluginV3OneBuild.get_output_shapes()`API 告知 TensorRT builder。表达输出的第二维非常直接：

```
# output_dims[0] = trt.DimsExprs(2)
output_dims[0][1] = exprBuilder.constant(2)
```

插件中每个数据依赖维度的“范围”必须用 **size tensor**（尺寸张量）来表达。尺寸张量是一个类型为 `trt.int32`或 `trt.int64`的标量输出，必须作为插件输出之一添加。在本例中，声明一个尺寸张量来表示非零索引输出第一维的范围就足够了。要声明尺寸张量，必须为其范围提供一个上界（upper-bound）和一个最优值（optimum value），两者均为 `IDimensionExpr`类型。这些可以通过传递给 `IPluginV3OneBuild.get_output_shapes()`方法的 `IExprBuilder`参数来构建。

- 对于未知输入，上界是输入中的元素总数：

  ```
  upper_bound = exprBuilder.operation(trt.DimensionOperation.PROD, inputs[0][0], inputs[0][1])
  ```

- 一个合理的优化估计值是约一半的元素为非零：

  ```
  opt_value = exprBuilder.operation(trt.DimensionOperation.FLOOR_DIV, upper_bound, exprBuilder.constant(2))
  ```

现在我们可以使用 `IExprBuilder.declare_size_tensor()`方法声明尺寸张量，该方法还需要指定尺寸张量所在的输出索引。将其放在非零索引输出之后：

```
num_non_zero_size_tensor = exprBuilder.declare_size_tensor(1, opt_value, upper_bound)
```

现在我们可以指定非零索引输出第一维的范围了：

```
# output_dims[0] = trt.DimsExprs(0)
output_dims[0][0] = num_non_zero_size_tensor
```

注意，尺寸张量被声明为一个标量（0-D）：

### 创建网络并构建 Engine

要将插件添加到网络中，必须使用 `INetworkDefinition::add_plugin_v3()`方法。

类似于 V2 插件使用的 `IPluginCreator`，V3 插件必须伴随一个实现了 `IPluginCreatorV3One`接口的插件创建器的注册。

## 运行样例

1. 运行样例以创建 TensorRT 推理 engine 并执行推理：

   `python3 non_zero_plugin.py [-h] [--precision {fp32,fp16}] [--backend {cuda_python,torch}] [--net_type {onnx,inetdef}]`

2. 验证样例是否成功运行。如果样例运行成功，您应该看到以下消息：

   ```
   Inference result correct!
   ```

### 样例 `--help`选项

要查看可用选项的完整列表及其描述，请使用 `-h`或 `--help`命令行选项。

# 附加资源

以下资源有助于更深入地理解 V3 TensorRT 插件和 NonZero 操作：

**NonZero**

- [ONNX: NonZero](https://onnx.ai/onnx/operators/onnx__NonZero.html)

**基于 C++ 的 NonZero 插件样例**

- [NonZero C++ Plugin](sampleNonZeroPlugin/)

**TensorRT 插件**

- [使用自定义层扩展 TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending)

- [基于 Python 的 TensorRT 插件](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/#add_custom_layer_python)

**其他文档**

- [NVIDIA TensorRT 样例简介](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)

- [使用 Python API 操作 TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/#python_topics)

- [NVIDIA TensorRT 文档库](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# 许可证

有关使用、复制和分发的条款和条件，请参阅 [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)文档。

# 变更日志

2025 年 10 月

迁移至强类型（strongly typed）API。

2025 年 8 月

移除对 < 3.10 Python 版本的支持。

2024 年 4 月

这是本 `README.md`文件的第一个版本。

# 已知问题

本样例中暂无已知问题。
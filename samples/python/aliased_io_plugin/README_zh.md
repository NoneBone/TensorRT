# 利用带别名 I/O 的插件实现原地更新

 en [English](./README.md) ｜ zh_CN [简体中文](./README_zh.md)

## 说明

本样例 `aliased_io_plugin`实现了一个基于 Python 的插件，用于原地（in-place）scatter-add 操作。

Scatter-add 根据给定的索引集合，将一组源值"分散"写入内存位置，并将映射到同一位置的值相加。

## 本样例如何工作？

本样例创建并运行一个 TensorRT engine，演示图神经网络（GNN, Graph Neural Networks）中常见的场景。在 GNN 中，每个节点的邻居特征通过一个与顺序无关的操作（如求和、乘积）聚合，再按邻域大小取平均，然后送入分类器以判定目标属性；GNN 的典型应用包括社交网络建模和推荐系统构建。

此处我们以加法作为聚合函数，因此构建一个包含 Scatter-add 插件节点的网络。它接收一个"源"张量（存放每个节点的邻居特征）和一个"索引"张量（表示每个邻居所属的节点索引）。例如，考虑下图：

aliased_io_gnn.png

为简化起见，本例及整个样例中每个节点使用标量特征。"源"可表示为展平张量 `[1.0, 3.0, 5.0, 7.0, 1.0, 3.0]`，对应的源节点索引为 `[1, 2, 3, 0, 2, 3]`。显然 Scatter-add 的结果应为 `[7.0, 1.0, 4.0, 8.0]`。该结果随后按每个节点的邻居数归一化，再送入一个简单的全连接层（dense layer）并接 ReLU 激活。

### 使用 `IPluginV3OneBuildV2`接口实现原地 Scatter-add 插件

在引入 `IPluginV3OneBuildV2`接口之前，TensorRT 插件输入被视为只读。由于这一限制，原地优化（输出写回某个输入）以及本质上需要修改输入的操作均无法实现。

在 Scatter-add 操作中，原地操作很有用，因为目标节点可能带有一些前置条件，要求将邻域聚合结果与偏置（bias）合并。另一个用例是层级聚合（hierarchical aggregation），其中更高层的特征也可能需要一并整合。

为了允许写入输入，`IPluginV3OneBuildV2`接口提供了一个 API，用于将某些输入-输出对声明为别名关系。在本例中，插件的第一个输出与第一个输入互为别名，因此可以这样声明：

```
def get_aliased_input(self, output_index: int):
	if output_index == 0:
		return 0

	return -1
```

返回值 `-1`表示该 `output_index`不与任何输入别名。

这一新方法 `get_aliased_input`是 `IPluginV3OneBuildV2`与 `IPluginV3OneBuild`的唯一区别。作为 `V3_ONE`能力接口集的一部分，`IPluginV3OneBuildV2`可与 `IPluginV3OneCore`和 `IPluginV3OneRuntime`配合使用。

### 创建网络并构建 Engine

要将插件加入网络，使用 `INetworkDefinition::add_plugin_v3()`方法。

后续的求平均和分类步骤，使用 TensorRT 的 ElementWise、MatrixMultiply、Activation 和 SoftMax 层完成。

## 运行样例

1. 

   运行样例以创建 TensorRT 推理 engine 并执行推理：

   `python3 aliased_io_plugin.py [-h] [--precision {fp32,fp16}] [--node_features NODE_FEATURES] [--edges EDGES] [--num_classes NUM_CLASSES] [--validate] [--seed SEED]`

2. 

   若传入了 `--validate`标志，验证样例是否运行成功。若运行成功，应看到如下信息：

   ```
   Validation against reference successful!
   ```

### 样例 `--help`选项

查看完整可用选项列表及说明，使用 `-h`或 `--help`命令行参数。

------

# 额外资源

以下资源有助于更深入理解 V3 TensorRT 插件及 Scatter-Add 操作：

**ScatterElements**

- [ONNX: ScatterElements](https://onnx.ai/onnx/operators/onnx__ScatterElements.html)

**TensorRT 插件**

- [用自定义层扩展 TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending)

- [TensorRT 基于 Python 的插件](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/#add_custom_layer_python)

**其他文档**

- [NVIDIA TensorRT 样例入门](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)

- [使用 Python API 操作 TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/#python_topics)

- [NVIDIA TensorRT 文档库](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

------

# 许可证

使用、复制和分发的相关条款与条件，请参见 [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)文档。

------

# 变更日志

- 2025 年 10 月：迁移至强类型（strongly typed）API。

- 2025 年 8 月：移除对 < 3.10 Python 版本的支持。

- 2024 年 8 月：本 `README.md`文件首次发布。

------

# 已知问题

本样例暂无已知问题。
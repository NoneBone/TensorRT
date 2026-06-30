# TopkLastDim

**目录**
- #description
    * #structure
- #parameters
- #additional-resources
- #license
- #changelog
- #known-issues

## 描述

`TopkLastDim` 插件用于计算输入张量沿指定轴的前 `k` 个最大或最小元素，其取值、排序逻辑以及 `axis` / `k` / `largest` 参数的含义遵循 https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK。该插件输出 `int32` 类型的索引，而非 ONNX 规范要求的 `int64` 索引——详见 #known-issues。

该插件基于从 https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/topkLastDim.cu 移植而来的 AIR (Adaptive Iterative Radix) 排序内核构建，该内核针对 2D `[numRows, rowLength]` 视图的最后一维进行操作。插件通过以下方式处理任意输入维度和任意轴：

- **快速路径**（`axis == 最后一维`）：直接在输入上调用内核，无需拷贝。
- **通用路径**（任何其他轴）：将输入进行转置，使目标轴移动到最后一维，在内核处理生成的 2D 视图后，再将输出的数值和索引转置回去，以确保前 `k` 维位于原始轴的位置。

### 结构

`TopkLastDim` 插件接收以下输入：

1. `input` - T：任意维度的张量。插件将沿 `axis` 属性指定的轴计算前 `k` 个元素。T 可以是 `float32`、`float16`、`int32` 或 `bfloat16`。

`TopkLastDim` 插件产生以下输出：

1. `values` - T：形状与 `input` 相同，除了沿 `axis` 的大小被替换为 `k`。包含沿 `axis` 的前 `k` 个值，当 `is_largest == 1` 时为降序排列，当 `is_largest == 0` 时为升序排列。
2. `indices` - `int32`：形状与 `values` 相同的张量。包含 `values` 中对应条目在 `input` 的 `axis` 轴上的索引。**注意**：ONNX TopK 规范要求 `int64` 索引；本插件输出 `int32` 索引。

该插件拥有插件创建器类 `TopkLastDimPluginCreator` 和继承自 `IPluginV3` 的插件类 `TopkLastDimPlugin`。

## 参数

`TopkLastDim` 插件包含以下参数：

| 类型    | 参数          | 描述
|---------|---------------|--------------------------------------------------------
| `int32` | `type_id`     | 输入张量的数据类型（也是 `values` 输出的类型）。允许的值遵循 `nvinfer1::DataType`：`0` (kFLOAT)、`1` (kHALF)、`3` (kINT32)、`7` (kBF16)。必填。
| `int32` | `k`           | 沿 `axis` 返回的最大元素数量。必须是正整数且不大于 `input` 沿 `axis` 的大小。必填。
| `int32` | `is_largest`  | 若为 `1`，返回最大的 `k` 个元素（降序排序）；若为 `0`，返回最小的 `k` 个元素（升序排序）。必填。
| `int32` | `axis`        | 计算前 `k` 个元素的轴。负值表示从后向前计数（例如，`-1` 表示最后一维）。可选；默认为 `-1`。

## 其他资源

以下资源有助于更深入地理解 `TopkLastDim` 插件：

- https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/topkLastDim.cu

## 许可证

有关使用、复制和分发的使用条款和条件，请参阅 https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html。

## 更新日志

2026 年 5 月：这是本 `README.md` 文件的首次发布。

## 已知问题

- ONNX TopK 规范要求索引类型为 `int64`；本插件输出 `int32` 索引，以匹配上游 TRT-LLM 内核的公共入口点（该入口点固定 `IdxT = int32_t`）。因此，沿 `axis` 长度超过 `2^31 - 1` 的输入将无法寻址。
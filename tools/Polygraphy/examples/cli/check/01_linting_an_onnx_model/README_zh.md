# 检查 ONNX 模型


## 介绍

这 `check lint` 子工具验证 ONNX 模型，并生成 JSON 报告，详细说明任何错误/未使用的节点或模型错误。

## 运行示例

### 检查 ONNX 模型：

<!-- Polygraphy Test: XFAIL Start -->
```bash
polygraphy check lint bad_graph.onnx -o report.json
```
<!-- Polygraphy Test: XFAIL End -->
输出结果应该类似于这样：
```bash
[I] RUNNING | Command: polygraphy check lint bad_graph.onnx -o report.json
[I] Loading model: bad_graph.onnx
[E] LINT | Field 'name' of 'graph' is required to be non-empty.
[I] Will generate inference input data according to provided TensorMetadata: {E [dtype=float32, shape=(1, 4)],
     F [dtype=float32, shape=(4, 1)],
     G [dtype=int64, shape=(4, 4)],
     D [dtype=float32, shape=(4, 1)],
     C [dtype=float32, shape=(3, 4)],
     A [dtype=float32, shape=(1, 3)],
     B [dtype=float32, shape=(4, 4)]}
[E] LINT | Name: MatMul_3, Op: MatMul |  Incompatible dimensions for matrix multiplication
[E] LINT | Name: Add_0, Op: Add |  Incompatible dimensions
[E] LINT | Name: MatMul_0, Op: MatMul |  Incompatible dimensions for matrix multiplication
[W] LINT | Input: 'A' does not affect outputs, can be removed.
[W] LINT | Input: 'B' does not affect outputs, can be removed.
[W] LINT | Name: MatMul_0, Op: MatMul | Does not affect outputs, can be removed.
[I] Saving linting report to report.json
[E] FAILED | Runtime: 1.006s | Command: polygraphy check lint bad_graph.onnx -o report.json
```

- 这将创建一个 `report.json` 其中包含有关模型出现问题的信息。
- 上述示例使用了有缺陷的 ONNX 模型 `bad_graph.onnx` 代码检查器捕获到多个错误/警告。
错误如下：
    1. 模型名称为空。
    2. 节点 `Add_0`， `MatMul_0` 和 `MatMul_3` 输入形状不兼容。
警告内容如下：
    1. 输入 `A` 和 `B` 未使用的输出。
    2. 节点 `MatMul_0` 未被输出使用。

### 示例报告：

生成的报告如下所示：

<!-- Polygraphy Test: Ignore Start -->
```json
{
    "summary": {
        "passing": [
            "MatMul_1",
            "cast_to_int64",
            "NonZero"
        ],
        "failing": [
            "MatMul_0",
            "MatMul_3",
            "Add_0"
        ]
    },
    "lint_entries": [
        {
            "level": "exception",
            "source": "onnx_checker",
            "message": "Field 'name' of 'graph' is required to be non-empty."
        },
        {
            "level": "exception",
            "source": "onnxruntime",
            "message": " Incompatible dimensions for matrix multiplication",
            "nodes": [
                "MatMul_3"
            ]
        },
        {
            "level": "exception",
            "source": "onnxruntime",
            "message": " Incompatible dimensions",
            "nodes": [
                "Add_0"
            ]
        },
        {
            "level": "exception",
            "source": "onnxruntime",
            "message": " Incompatible dimensions for matrix multiplication",
            "nodes": [
                "MatMul_0"
            ]
        },
        {
            "level": "warning",
            "source": "onnx_graphsurgeon",
            "message": "Input: 'A' does not affect outputs, can be removed."
        },
        {
            "level": "warning",
            "source": "onnx_graphsurgeon",
            "message": "Input: 'B' does not affect outputs, can be removed."
        },
        {
            "level": "warning",
            "source": "onnx_graphsurgeon",
            "message": "Does not affect outputs, can be removed.",
            "nodes": [
                "MatMul_0"
            ]
        }
    ]
}
```
<!-- Polygraphy Test: Ignore End -->

### 笔记
由于它在底层运行 ONNX Runtime，因此可以使用以下方式指定执行提供程序： `--providers`默认使用 CPU。

也可以使用以下方法覆盖输入形状 `--input-shapes`或者提供自定义输入数据。更多详情，请参阅 [how-to/use_custom_input_data](../../../../how-to/use_custom_input_data.md)。

有关使用方法的更多信息，请使用 `polygraphy check lint --help`。

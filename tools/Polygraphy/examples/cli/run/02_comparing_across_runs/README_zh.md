# 跨批次比较

## 先决条件
关于如何使用的总体概述 `polygraphy run` 比较以下输出：
不同的框架，请参见示例。 [Comparing Frameworks](../../../../examples/cli/run/01_comparing_frameworks)。

## 介绍

有些情况下，你可能需要比较不同调用方式的结果。
的 `polygraphy run` 命令。例如：

* 比较不同平台的结果
* 比较不同版本 TensorRT 的结果
* 比较具有兼容输入/输出的不同型号

在这个例子中，我们将演示如何使用测谎技术来实现这一点。

## 运行示例

### 跨批次比较

1. 保存第一次运行的输入和输出值：

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --save-inputs inputs.json --save-outputs run_0_outputs.json
    ```

2. 再次运行模型，这次加载已保存的输入和输出。
    第一次运行。保存的输入将用作本次运行的输入，并且
    保存的输出结果将与第一次运行的结果进行比较。

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --load-inputs inputs.json --load-outputs run_0_outputs.json
    ```

    这 `--atol/--rtol/--check-error-stat` 所有选项的功能都与之前相同。
    [Comparing Frameworks](../../../../examples/cli/run/01_comparing_frameworks) 例子：

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --load-inputs inputs.json --load-outputs run_0_outputs.json \
        --atol 0.001 --rtol 0.001 --check-error-stat median
    ```

### 不同型号的比较

我们还可以使用这种技术来比较不同的模型，例如 TensorRT 引擎。
以及 ONNX 型号（如果它们有匹配的输出）。

1. 将 ONNX 模型转换为 TensorRT 引擎并保存到磁盘：

    ```bash
    polygraphy convert identity.onnx -o identity.engine
    ```

2. 在 Polygraphy 中运行已保存的引擎，使用从 ONNX-Runtime 运行中保存的输入作为参数。
    将引擎的输入与保存的 ONNX-Runtime 输出进行比较：

    ```bash
    polygraphy run --trt identity.engine --model-type=engine \
        --load-inputs inputs.json --load-outputs run_0_outputs.json
    ```


## 延伸阅读

有关如何访问和使用已保存的输出的详细信息
使用 Python API 时，请参考 [API example 08](../../../api/08_working_with_run_results_and_saved_inputs_manually/)。

有关与自定义输出进行比较的信息，请参阅 [`run` example 06](../06_comparing_with_custom_output_data/)。

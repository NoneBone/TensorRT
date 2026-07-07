# 减少 ONNX 模型故障

## 介绍

当模型因任何原因失效时（例如，TensorRT 中的准确率问题），通常是
将其简化为触发故障的最小子图很有用。这使得
这样更容易找出故障原因。

实现这一目标的一种方法是生成原始 ONNX 模型的越来越小的子图。
在每次迭代中，我们可以检查子图是否有效或仍然无效；一旦我们得到一个有效的子图，就可以继续进行后续步骤。
子图，我们知道由前一次迭代生成的子图是最小的失败子图
子图。

这 `debug reduce` 子工具使我们能够自动执行此过程。


## 运行示例

为了便于举例，我们假设我们的模型（`./model.onnx`存在准确性问题
在 TensorRT 中。由于该模型实际上在 TensorRT 中可以正常工作（如果不行，请报告错误！），
我们将概述您通常会运行的命令，以及您可以运行的命令。
模拟故障情况，以便了解该工具在实际使用中的表现。

只要出现以下情况，我们的模拟故障就会触发： `Mul` 模型中的节点：

![./model.png](./model.png)

因此，最终的简化模型应该只包含 `Mul` 节点（因为其他节点不会导致故障）。

1. 对于使用动态输入形状或包含形状操作的模型，冻结输入
    使用以下方式进行形状和折叠形状操作：

    ```bash
    polygraphy surgeon sanitize model.onnx -o folded.onnx --fold-constants \
        --override-input-shapes x0:[1,3,224,224] x1:[1,3,224,224]
    ```

2. 假设 ONNX-Runtime 能给出正确的输出。我们首先生成黄金标准。
    网络中每个张量的值。我们还会保存所使用的输入：

    ```bash
    polygraphy run folded.onnx --onnxrt \
        --save-inputs inputs.json \
        --onnx-outputs mark all --save-outputs layerwise_golden.json
    ```

    然后我们将输入和逐层输出合并到一个逐层输入文件中。
    使用 `data to-input` 子工具（我们将在下一步中看到为什么需要它）：

    ```bash
    polygraphy data merge inputs.json layerwise_golden.json -o layerwise_inputs.json
    ```


3. 接下来，我们将使用 `debug reduce` 在 `bisect` 模式：

    ```bash
    polygraphy debug reduce folded.onnx -o initial_reduced.onnx --mode=bisect --load-inputs layerwise_inputs.json \
        --check polygraphy run polygraphy_debug.onnx --trt \
                --load-inputs layerwise_inputs.json --load-outputs layerwise_golden.json
    ```

    让我们来详细分析一下：

    - 和其他 `debug` 子工具， `debug reduce` 每次迭代都会生成一个中间产物
        （`./polygraphy_debug.onnx` 默认情况下）。在这种情况下，工件是原始 ONNX 模型的某个子图。

    - 为了 `debug reduce` 确定每个子图是否失败或通过，
        我们提供一种 `--check` 命令。由于我们正在调查准确性问题，
        我们可以使用 `polygraphy run` 与我们之前的最佳结果进行比较。

        *提示：和其他 `debug` 子工具还支持交互模式，您可以*
            *只需省略即可 `--check` 争论。*

    - 在 `--check` 通过命令，我们提供逐层输入。 `--load-inputs`否则， `polygraphy run`
        这将为子图张量生成新的输入，这些输入可能与这些张量的值不匹配。
        我们在生成黄金数据时就已经这样做了。另一种方法是运行参考实现。
        （此处为 ONNX-Runtime）在每次迭代期间 `debug reduce` 而不是提前准备。

    - 由于我们使用的是非默认输入数据，因此我们也通过以下方式提供逐层输入： `--load-inputs` 直接到
        `debug reduce` 命令（除了将其提供给 `--check` 命令）。
        这在具有多个并行分支的模型中非常重要（*这里指的是模型中的路径，而不是控制流*），例如：
        <!-- Polygraphy Test: Ignore Start -->
        ```
         inp0  inp1
          |     |
         Abs   Abs
            \ /
            Sum
             |
            out
        ```
        在这种情况下， `debug reduce` 需要能够用常量替换一个分支。
        为此，它需要知道你正在使用的输入数据，以便将其替换为正确的值。
        虽然这里使用的是文件，但输入数据也可以通过 Polygraphy 数据加载器中介绍的任何其他参数提供。
        [the CLI user guide](../../../../how-to/use_custom_input_data.md)。

        如果您不确定是否需要提供数据加载器，
        `debug reduce` 当尝试替换分支时，会发出类似这样的警告：
        ```
        [W]     This model includes multiple branches/paths. In order to continue reducing, one branch needs to be folded away.
                Please ensure that you have provided a data loader argument to `debug reduce` if your `--check` command is using a non-default data loader.
                Not doing so may result in false negatives!
        ```
        <!-- Polygraphy Test: Ignore End -->

    - 我们明确规定 `-o` 选择此选项，以便将简化模型写入 `initial_reduced.onnx`。

    **模拟故障：**我们可以使用 `polygraphy inspect model` 与 `--fail-regex` 触发
    当模型包含以下内容时，就会发生故障 `Mul` 节点：

    ```bash
    polygraphy debug reduce folded.onnx -o initial_reduced.onnx --mode=bisect \
        --fail-regex "Op: Mul" \
        --check polygraphy inspect model polygraphy_debug.onnx --show layers
    ```

4. **[可选]** 作为健全性检查，我们可以检查简化后的模型，以确保它确实包含 `Mul` 节点：

    ```bash
    polygraphy inspect model initial_reduced.onnx --show layers
    ```

5. 由于我们使用 `bisect` 如果采用上一步的模式，则模型可能不够简洁。
    为了进一步完善它，我们将运行 `debug reduce` 再次 `linear` 模式：

    ```bash
    polygraphy debug reduce initial_reduced.onnx -o final_reduced.onnx --mode=linear --load-inputs layerwise_inputs.json \
        --check polygraphy run polygraphy_debug.onnx --trt \
                --load-inputs layerwise_inputs.json --load-outputs layerwise_golden.json
    ```

    **模拟故障：** 我们将使用与之前相同的技术：

    ```bash
    polygraphy debug reduce initial_reduced.onnx -o final_reduced.onnx --mode=linear \
        --fail-regex "Op: Mul" \
        --check polygraphy inspect model polygraphy_debug.onnx --show layers
    ```

6. **[可选]** 在此阶段， `final_reduced.onnx` 应该只包含故障节点—— `Mul`。
    我们可以通过以下方式验证这一点： `inspect model`：

    ```bash
    polygraphy inspect model final_reduced.onnx --show layers
    ```


## 延伸阅读

- 有关如何操作的更多详细信息 `debug` 工具运行正常，请查看帮助输出：
    `polygraphy debug -h` 和 `polygraphy debug reduce -h`。

- 另请参阅 [`debug reduce` how-to guide](../../../../how-to/use_debug_reduce_effectively.md)
    了解更多信息、技巧和窍门。

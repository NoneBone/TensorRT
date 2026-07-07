# 调试不稳定的 TensorRT 策略

**重要提示：由于 TensorRT 版本更新，此示例在新版本中已不再可靠运行。**
    **未通过 IAlgorithmSelector 接口公开的策略选择（TensorRT 10.8 中已弃用）。**
    （请改用 ITimingCache 中的可编辑模式）。因此，采用以下概述的方法。
    **无法保证引擎构建的确定性。使用 TensorRT 8.7 及更高版本，您可以使用**
    战术计时缓存（`--save-timing-cache` 和 `--load-timing-cache` 在测谎中）以确保**
    **确定性，但这些文件是不透明的，因此无法被解释 `inspect diff-tactics`**

## 介绍

有时，TensorRT 中的某种策略可能会产生错误的结果，或者存在其他问题。
否则会出现错误行为。因为 TensorRT 构建器依赖于时间信息。
战术和引擎构建是不确定的，这可能会导致战术漏洞。
表现为不稳定/间歇性故障。

解决该问题的一种方法是多次运行构建器，
保存每次运行的战术回放文件。一旦我们有了一组已知有效且
已知的不良战术，我们可以进行比较，以确定哪种战术最有效。
很可能是误差的来源。

这 `debug build` 子工具允许您自动执行此过程。

有关如何操作的更多详细信息 `debug` 工具运行正常，请查看帮助输出：
`polygraphy debug -h` 和 `polygraphy debug build -h`。


## 运行示例

1. 从 ONNX-Runtime 生成黄金输出：

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --save-outputs golden.json
    ```

2. 使用 `debug build` 反复构建TensorRT引擎，并将结果与​​黄金输出进行比较，
    每次都保存一个战术回放文件：

    ```bash
    polygraphy debug build identity.onnx --fp16 --save-tactics replay.json \
        --artifacts-dir replays --artifacts replay.json --until=10 \
        --check polygraphy run polygraphy_debug.engine --trt --load-outputs golden.json
    ```

    让我们来详细分析一下：

    - 像其他 `debug` 子工具， `debug build` 每次迭代都会生成一个中间产物
        （`./polygraphy_debug.engine` 默认情况下）。在这种情况下，该组件是一个 TensorRT 引擎。

        *提示： `debug build` 支持所有受支持的 TensorRT 构建器配置选项*
            *通过其他工具，例如 `convert` 或者 `run`.*

    - 为了 `debug build` 确定每台发动机是否合格或不合格，
        我们提供一种 `--check` 命令。既然我们正在讨论的是一个（虚假的）准确性问题，
        我们可以使用 `polygraphy run` 将发动机的输出与我们的黄金标准进行比较。

        *提示：和其他 `debug` 子工具还支持交互模式，您可以*
            *只需省略即可 `--check` 争论。*

    - 与其他不同 `debug` 子工具， `debug build` 没有自动终止条件，所以我们需要
        提供 `--until` 此选项用于让工具知道何时停止。它可以是一个数字。
        迭代次数，或 `"good"` 或者 `"bad"`在后一种情况下，该工具会在找到目标后停止。
        分别是第一次迭代通过或失败。

    - 由于我们最终想要比较好战术和坏战术的回放，我们特此说明 `--save-tactics`
        保存每次迭代的战术回放文件，然后使用 `--artifacts` 告诉 `debug build`
        为了管理它们，需要将它们分类 `good` 和 `bad` 子目录
        主工件目录，由以下方式指定 `--artifacts-dir`。


3. 使用 `inspect diff-tactics` 判断哪些策略可能无效：

    ```bash
    polygraphy inspect diff-tactics --dir replays
    ```

    *注：最后一步应报告无法确定潜在的不良策略，因为*
        *我们的 `bad` 此时目录应该为空（否则请提交 TensorRT 问题！）：*

    <!-- Polygraphy Test: Ignore Start -->
    ```
    [I] Loaded 2 good tactic replays.
    [I] Loaded 0 bad tactic replays.
    [I] Could not determine potentially bad tactics. Try generating more tactic replay files?
    ```
    <!-- Polygraphy Test: Ignore End -->


## 延伸阅读

有关更多信息 `debug` 工具，以及适用的技巧和窍门
致所有人 `debug` 子工具，请参阅
[how-to guide for `debug` subtools](../../../../how-to/use_debug_subtools_effectively.md)。

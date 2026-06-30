# 快速可部署的 TensorRT Python 插件［实验性］

本示例展示了如何在 TensorRT (TRT) 中使用基于 Python 的快速可部署插件定义（QDP，Quickly Deployable Python Plugins）。QDP 能够覆盖向 TRT 添加自定义算子的绝大多数使用场景，待 10.9 版本成为稳定特性后将作为推荐方案。

本示例包含若干迷你样例，演示几种常见使用场景。

# 目录

- [简介](#introduction)

- [环境配置](#setting-up-the-environment)

- [实现一个快速可部署 Python (QDP) 插件](#implementing-a-quickly-deployable-python-qdp-plugin)

- [简单插件：逐元素加法](#a-simple-plugin-elementwise-add)

- [通过 I/O 别名实现原地（in-place）自定义算子](#implementing-in-place-custom-ops-with-io-aliasing)

- [输出形状依赖数据的算子：Non-zero](#an-op-with-data-dependent-output-shapes-non-zero)

- [多 tactic 与 ONNX：环形填充（Circular Padding）](#using-multiple-tactics-and-onnx-cirular-padding)

- [为环形填充提供 AOT（Ahead-of-Time）实现](#poviding-an-ahead-of-time-aot-implementation-for-cirular-padding)

- [额外资源](#additional-resources)

- [许可证](#license)

- [变更日志](#changelog)

- [已知问题](#known-issues)

------

# 简介

虽然常规 TRT 插件接口在灵活性和可调优性上非常强大，但对绝大多数使用场景而言，用户会从 QDP 工作流的简洁性中获益：

- `tensorrt.plugin`模块提供了许多直观 API，大幅减少了实现插件所需的样板代码量；

- 插件注册、插件创建器（plugin creator）和插件注册表（plugin registry）的概念被抽象掉了；

- QDP 的无状态特性消除了必须遵守预定义插件生命周期的复杂性。

------

# 环境配置

构建并安装绑定，请遵循 `$TRT_OSSPATH/python/README.md`中的说明。

然后安装所需包：

```
cd $TRT_OSSPATH/samples/python/quickly_deployable_plugins
pip3 install -r requirements.txt
```

------

# 实现一个快速可部署 Python (QDP) 插件

QDP 定义由一组装饰器函数组成，用于描述插件的属性和行为。

### `@tensorrt.plugin.register`

返回输出张量的形状和类型特征，以及插件运行所需的属性。

### `@tensorrt.plugin.impl`

执行插件计算。被装饰的 Python 函数在运行时作为 Python 回调被"即时"（just in time）执行。

### （可选）`@tensorrt.plugin.aot_impl`

被装饰的函数直接返回一个"Ahead-of-Time"编译好的 kernel，以及 TRT 在运行时调用该 kernel 所需的信息。与上方的 `@tensorrt.plugin.impl`不同，返回的 kernel 会被烘焙进构建好的 TRT engine 中。当我们需要一个完全独立于 Python 运行时的 engine 时会很有用——因此可以在标准 TensorRT C++ 运行时中单独执行（例如通过 `trtexec`）。

### （可选）`@tensorrt.plugin.autotune`

定义插件 IO 支持的数据类型和格式（张量布局），以及插件支持的任意 tactic。定义此函数可让 TensorRT 在 engine 构建期间对插件进行"调优"，以找到目标系统上性能最佳的类型/格式与 tactic 组合。

上述函数的具体用法将通过以下迷你样例阐明。

------

# 简单插件：逐元素加法（Elementwise-Add）

本迷你样例包含一个逐元素加法插件，计算通过 OpenAI Triton kernel 完成。先来看 `tensorrt.plugin.register`函数：

```
import tensorrt.plugin as trtp

@trtp.register("sample::elemwise_add_plugin")
def add_plugin_desc(inp0: trtp.TensorDesc, block_size: int) -> trtp.TensorDesc:
    return inp0.like()
```

参数 `"sample::elemwise_add_plugin"`定义了插件的命名空间（`"sample"`）和名称（`"elemwise_add_plugin"`）。被装饰函数（`plugin_desc`）的输入参数中，用 `trt.plugin.TensorDesc`注解的表示输入张量；其余参数被解释为插件属性（完整允许的属性类型列表参见 [TRT API 参考](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/tensorrt.plugin/trt_plugin_register.html)）。输出签名是一个 `trt.plugin.TensorDesc`，描述输出。`inp0.like()`返回一个与 `inp0`形状和类型特征完全相同的张量描述符。

用 `trt.plugin.impl`装饰的计算函数，为每个输入和输出接收 `trt.plugin.Tensor`。与 `TensorDesc`不同，`Tensor`引用底层数据缓冲区，可通过 `Tensor.data_ptr`直接访问。配合 Torch 和 OpenAI Triton kernel 使用时，用 `torch.as_tensor()`以零拷贝方式构造对应 `trt.plugin.Tensor`的 `torch.Tensor`会更方便。

本样例还展示了省略/定义 `trt.plugin.autotune`函数的效果，该函数必须返回 `trt.plugin.AutoTuneCombination`列表。本例中我们定义了一个组合 `AutoTuneCombination("FP32|FP16, FP32|FP16")`，表示输入和输出必须同为 FP32 或同为 FP16。关于 `AutoTuneCombination`的语法详细说明，请参见 TRT API 参考。

## 运行样例

```
python3 qdp_runner.py add [--autotune] [-v] # TODO: 待修复
```

`--autotune`模拟已定义 `trt.plugin.autotune`函数的情况。建议开启详细日志（`-v`）观察 autotuning 的效果。可以观察到，启用 autotune 时，`trt.plugin.impl`函数在 engine 构建过程中会被多次调用；关闭 autotuning 时，`trt.plugin.impl`仅在 engine 构建完成后执行推理时被调用一次。

```
$ python3 qdp_runner.py non_zero -v
```

------

# 通过 I/O 别名实现原地（in-place）自定义算子

原地计算可以通过带别名的 I/O 在 TRT 插件中实现，即：需要被原地修改的输入可以用一个"输入输出对"来表示，其中输出与输入别名相同。例如，如果需要原地加法（而非上述样例中的非原地加法），可以这样实现：

```
import tensorrt.plugin as trtp

@trtp.register("sample::elemwise_add_plugin_")
def add_plugin_desc_(inp0: trtp.TensorDesc) -> trtp.TensorDesc:
    return inp0.aliased()
```

注意这里用了 `trt.plugin.TensorDesc.aliased()`来生成一个与 `inp0`别名相同的输出 `TensorDesc`。

为了更好地体会别名的效果，本样例串联了两个原地加法插件。

## 运行样例

建议开启详细日志（`-v`）观察始终启用的 autotuning 效果。

```
python3 qdp_runner.py inplace_add [--autotune] [-v]
```

------

# 输出形状依赖数据的算子：Non-zero

Non-zero 操作用于找出输入张量中非零元素的坐标——它具有**数据依赖的输出形状**（DDS, Data-Dependent Shapes）。因此，常规的基于输入形状的形状计算无法处理这种情况。

为处理 DDS，每个数据依赖的输出维度的"范围"必须用 **size tensor**（尺寸张量）来表达：它是一个标量，向 TRT 传达该维度在输入形状意义下的上界（upper bound）和调优值（autotune value）。TRT engine 构建可以针对 autotune 值优化，但该维度的实际范围在运行时最多可拉伸到上界。

本样例中，我们考虑一个 2D 输入张量 `inp0`；输出将是一个 N×2的张量（N个二维坐标），其中 N是非零元素个数。最坏情况下所有元素都非零，因此上界可表示为 `upper_bound = inp0.shape_expr[0] * inp0.shape_expr[1]`。注意 `trt.plugin.TensorDesc.shape_expr`返回该张量的符号形状表达式，形状表达式上的算术运算通过标准 Python 二元运算符支持（完整支持操作见 [TRT Python API 参考](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/tensorrt.plugin/Shape/ShapeExpr.html)）。

平均而言，可以预期输入约一半为零，因此可以用此作为 autotune 值构造 size tensor：

```
st = trtp.size_tensor(opt = upper_bound // 2, upper_bound = upper_bound)
```

接下来就可以构造输出形状了。`st.expr()`返回 size tensor 对应的形状表达式，因此输出的张量描述符可构造为 `trt.plugin.from_shape_expr((st.expr(), 2), dtype=trt.int32)`。TRT 要求所有 size tensor 也必须作为插件的输出。整合起来得到如下代码：

```
import tensorrt.plugin as trtp

@trtp.register("sample::non_zero_plugin")
def non_zero_plugin_reg(
    inp0: trtp.TensorDesc,
) -> Tuple[trtp.TensorDesc, trtp.TensorDesc]:
    upper_bound = inp0.shape_expr[0] * inp0.shape_expr[1]
    st = trtp.size_tensor(upper_bound // 2, upper_bound)
    return trtp.from_shape_expr((st.expr(), 2), dtype=trt.int32), st
```

## 运行样例

建议开启详细日志（`-v`）观察始终启用的 autotuning 效果。

```
python3 qdp_runner.py non_zero [-v]
```

------

# 多 tactic 与 ONNX：环形填充（Circular Padding）

本样例包含一个环形填充插件，对环形卷积等算子有用，等价于 PyTorch 的 [torch.nn.CircularPad2d](https://pytorch.org/docs/stable/generated/torch.nn.CircularPad2d.html#torch.nn.CircularPad2d)。

更多信息参见 Python 插件指南中[关于环形填充插件的章节](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/pluginGuide.html#example-circular-padding-plugin)。

## 带插件的 ONNX 模型

通过你自己写的 TRT 插件运行含 custom op 的 ONNX 节点通常很有用。为了让 TRT ONNX parser 正确识别你的插件映射到某个 ONNX 节点，需确保：

- 节点的 `op`属性与你的插件名称完全一致；

- 节点包含一个名为 `"plugin_namespace"`的字符串属性，值为你的插件命名空间。

本样例中，我们定义了一个 ID 为 `"sample::circ_pad_plugin"`的插件，因此如果用 ONNX Graphsurgeon，custom op 节点可这样构造：

```
import onnx_graphsurgeon as gs

var_x = gs.Variable(name="x", shape=inp_shape, dtype=np.float32)
var_y = gs.Variable(name="y", dtype=np.float32)

circ_pad_node = gs.Node(
    name="circ_pad_plugin",
    op="circ_pad_plugin",
    inputs=[var_x],
    outputs=[var_y],
    attrs={"pads": pads, "plugin_namespace": "sample"},
)
```

## 多 tactic

有时你可能有多个 kernel（或后端）可用于执行插件计算——这些通常称为 **tactic**。如果无法预先确定哪个 tactic 最快，可以让 TRT 对每个 tactic 计时并由 TRT 决定最快的那个。

通过 `trt.plugin.autotune`函数即可告知 TRT 存在多个 tactic：

```
import tensorrt.plugin as trtp
from enum import IntEnum

class Tactic(IntEnum):
    TORCH = 1
    TRITON = 2

@trt.plugin.autotune("sample::circ_pad_plugin")
def circ_pad_plugin_autotune(inp0: trtp.TensorDesc, pads: npt.NDArray[np.int32], outputs: Tuple[trtp.TensorDesc]) -> List[trtp.AutoTuneCombination]:
    c = trtp.AutoTuneCombination()
    c.pos([0, 1], "FP32|FP16")
    c.tactics([int(Tactic.TORCH), int(Tactic.TRITON)])
    return [c]
```

注意这里我们用了另一种构造 `trt.plugin.AutoTuneCombination`的方式——即通过 `pos(...)`填入类型/格式信息，通过 `tactics(...)`指定 tactic。本样例中，我们使用 OpenAI Triton kernel 和 `torch.nn.functional.pad`作为计算环形填充的两种方法。

更多信息参见 Python 插件指南中[关于多后端自定义 tactic 的章节](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/pluginGuide.html#example-plugins-with-multiple-backends-using-custom-tactics)。

## 加载并运行包含插件的 TRT Engine

如果你有一个用插件构建好的 TRT engine，执行该 engine 只需要在反序列化 engine 的模块中提供 `trt.plugin.register`和 `trt.plugin.impl`的插件定义（注意：`trt.plugin.autotune`定义不必存在）。

为模拟 engine 加载，先用 `--save_engine`标志运行本样例，并通过 `--artifacts_dir [dir]`指定希望保存 engine 的目录；然后再用 `--load engine`和同样的 `--artifacts_dir`再次运行样例。

## 运行样例

```
python3 qdp_runner.py circ_pad [--multi_tactic] [--save_engine] [--load_engine] --mode {onnx,inetdef} [--artifacts_dir ARTIFACTS_DIR]  [-v]

options:
  --multi_tactic       启用多 tactic。
  --save_engine        将 engine 保存到 artifacts_dir。
  --load_engine        从 artifacts_dir 加载 engine，忽略其他选项。
  --artifacts_dir ARTIFACTS_DIR
                       是否存储（或读取）产物。
  --mode {onnx,inetdef}
                       使用 ONNX parser 还是 INetworkDefinition API 构建网络。
  -v, --verbose        启用详细日志输出。
```

------

# 为环形填充提供 AOT（Ahead-of-Time）实现

我们在[上述样例](#using-multiple-tactics-and-onnx-cirular-padding)基础上扩展，为同一个环形填充操作提供 AOT 实现。不再通过 `@trt.plugin.impl`向 TRT 指定 OpenAI Triton kernel 回调，而是可以直接提前编译好 kernel，并在 `@trt.plugin.aot_impl`下提供给 TRT。

更多信息参见 Python 插件指南中[关于 AOT 实现的章节](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/pluginGuide.html#providing-an-ahead-of-time-aot-implementation)。

## 带 AOT 插件的 ONNX 模型

规则与上述[带插件的 ONNX 模型](#onnx-model-with-a-plugin)章节相同。此外，如果插件有我们希望使用的 AOT 实现，可以修改 ONNX 节点以告知 TRT ONNX parser——需在 ONNX 节点上增加一个 bool 属性 `"aot"`并设为 `True`。注意这是在确保 ONNX 节点具备正确的 `op`属性和 `"plugin_namespace"`属性（[前文](#onnx-model-with-a-plugin)已提及）之上的额外步骤。

因此，使用 ONNX Graphsurgeon 时，使用 `"sample::circ_pad_plugin"`的 AOT 实现的 custom op 节点可类似构造：

```
import onnx_graphsurgeon as gs

var_x = gs.Variable(name="x", shape=inp_shape, dtype=np.float32)
var_y = gs.Variable(name="y", dtype=np.float32)

circ_pad_aot_node = gs.Node(
    name="circ_pad_plugin_aot",
    op="circ_pad_plugin",
    inputs=[var_x],
    outputs=[var_y],
    attrs={"pads": pads, "plugin_namespace": "sample", "aot": True},
)
```

## 加载并运行包含 AOT 插件的 TRT Engine

如果你有一个用 AOT 插件构建的 TRT engine，插件计算已经是 engine 的一部分，因此运行时不需要任何 Python 模块或定义存在。这意味着该 engine 可以在标准 TRT 运行时中执行，作为任何能够反序列化并运行 engine 的工具（如 [trtexec](trtexec/README.md)）的一部分。

为模拟 engine 加载，先用 `--save_engine`标志运行本样例，并通过 `--artifacts_dir [dir]`指定保存目录；然后再用 `--load engine`和同样的 `--artifacts_dir`再次运行。

## 运行样例

```
python3 qdp_runner.py circ_pad [--save_engine] [--load_engine] --mode {onnx,inetdef} [--artifacts_dir ARTIFACTS_DIR]  [-v]

options:
  --save_engine        将 engine 保存到 artifacts_dir。
  --load_engine        从 artifacts_dir 加载 engine，忽略其他选项。
  --artifacts_dir ARTIFACTS_DIR
                       是否存储（或读取）产物。
  --mode {onnx,inetdef}
                       使用 ONNX parser 还是 INetworkDefinition API 构建网络。
  --aot                使用插件的 AOT 实现。
  -v, --verbose        启用详细日志输出。
```

------

# 额外资源

**Python 插件指南**

- [pluginGuide.md](documentation/python/pluginGuide.md)

**`tensorrt.plugin`API 参考**

- [`tensorrt.plugin`模块 API 参考](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/tensorrt.plugin/index.html)

**开发者指南**

- [用自定义层扩展 TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending)

------

# 许可证

使用、复制和分发的相关条款与条件，请参见 [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)文档。

------

# 变更日志

- 2025 年 10 月：迁移至强类型（strongly typed）API。

- 2025 年 8 月：移除对 < 3.10 Python 版本的支持。

- 2024 年 12 月：新增 AOT 插件章节，增加目录。

- 2024 年 10 月：本样例首次发布。

------

# 已知问题

本样例暂无已知问题。
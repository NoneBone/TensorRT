# 基于 Python 的 TRT 插件

这是一个展示 TRT 中基于 Python 的插件定义的示例。交付此功能并未更改现有的 TRT API，因此使用更新后的绑定不应破坏任何现有代码。

## 简介

直到 TRT 9.1，插件实现只能通过 TRT C++ API 完成。要在 Python 应用中使用插件，必须：

- 用 C++ 实现插件并构建成共享库
- 加载插件库并注册插件创建器（静态或动态）
- 检索插件创建器并通过相应的 Python API 创建插件实例

在创建绑定时遵循了以下设计考虑：

- 在 TensorRT 中实现、集成和运行插件无需额外的 C++ 代码
- 提供灵活性，可通过任何选择的方法实现插件的核函数
  - 许多库应运而生，通过 AOT/JIT 编译提供 CUDA 核函数支持
    - Numba、OpenAI Triton、CuPy 等
  - 甚至可以不显式编写核函数（例如利用 PyTorch 的功能性算子）
- 仅支持基于 `IPluginV2DynamicExt` 和 `IPluginV3` 的插件
  - 自 TRT 8.5 起，其他插件接口（除 `IPluginV2IOExt` 外）均已弃用

借助这些绑定，可以完全使用 Python 将插件实现并集成到 TRT 中。

## 设置构建环境

要构建并安装绑定，请按照 `$TRT_OSSPATH/python/README.md` 中的说明进行操作。

然后安装所需的包

```
cd $TRT_OSSPATH/samples/python/trt_python_plugin
pip3 install -r requirements.txt
```

如果在 CUDA 11.x 环境下测试，请安装 `cupy-cuda11x`。

# TensorRT Python 插件 API

在 Python 中实现 TRT 插件与在 C++ 中类似，需要实现 `IPluginV2DynamicExt`+`IPluginCreator` 或 `IPluginV3`+`IPluginCreatorV3One`。请参阅 TensorRT Python API 参考以获取简要说明。

## `IPluginV2DynamicExt` 的 C++ 和 Python API 差异

Python 中的接口方法与 C++ 对应方法在 API 上大多相似，除了 `serialize()` 和 `enqueue()`。

- C++ 的 `serialize()` API 是 `void serialize (void *buffer)`，插件向传入的 `buffer` 写入数据；而 Python API 是 `serialize(self) -> bytes`，该方法预期返回一个包含插件对象序列化表示的字节对象。
- 在 `enqueue()` 中，输入和输出张量的设备指针以其 `intptr_t` 转换形式传递。由于这些缓冲区由 TRT 创建和拥有，从 Python 端写入时需格外小心。
- 尚未提供非纯虚函数 `attachToContext()` 和 `detachFromContext()` 的绑定。

# 运行示例：循环填充插件

本示例包含一个循环填充插件，其中 `enqueue` 已通过多种框架实现，用于编写核函数或执行 GPU 算子（torch）。

每个脚本都接受一个命令行参数来选择精度（FP32 或 FP16）。例如：

```
python3 circ_pad_plugin_cuda_python.py --precision fp32 # fp32 或 fp16
```

## 循环填充

循环填充对于深度学习中的循环卷积等操作非常有用。下图展示了原始图像（红色）如何进行一次（绿色）和两次（蓝色）循环填充：

!circ_pad_example.png "Circular padding example"

该插件具有以下特征：

- 输入：4 维输入（例如 NxCxHxW）
- 属性：m 维参数 `pads`，其中 m 为偶数且 m/2 \le 4。`pads` 表示在输入张量最后 m/2 个维度的前后分别应用的填充量。
- 输出：填充后的张量。形状取决于 `pads`。

## 基准测试：使用 C++ 插件

为了建立基准，我们首先演示一个实现循环填充的 C++ 插件。相关文件可在 `circ_plugin_cpp` 文件夹中找到：包含的 `CMakeLists.txt` 可用于构建共享库 `libcirc_pad_plugin.so` / `circ_pad_plugin.dll`。

```
cd $TRT_OSSPATH/samples/python/trt_python_plugin
mkdir build && pushd build
cmake .. && make -j
popd
python3 circ_pad_plugin_cpp.py --plugin-lib build/libcirc_pad_plugin.so
```

## Python 插件：cuda-python

基于 cuda-python 的实现可在 `circ_pad_plugin_cuda_python.py` 中找到。使用 `cuda.nvrtc` 对基于 C/C++ 的核函数进行 JIT 编译，核函数以字符串形式提供。编译后的核函数通过 cuda-python 的 `cuda.cuLaunchKernel` 启动。

`circ_pad_plugin_cuda_python.py` 演示了基于 ONNX 的工作流：`circ_pad_plugin_inetdef_cuda_python.py` 演示了通过 `INetworkDefinition` 构建模型的工作流。

## Python 插件：CuPy

基于 CuPy 的实现可在 `circ_pad_plugin_cupy.py` 中找到。使用了 CuPy 的 `RawKernel` 类来提供基于 C/C++ 的核函数实现（以字符串形式）。CuPy 将对核函数进行 JIT 编译。

## Python 插件：Triton（仅在 Linux 上有效）

同样的插件也可以用基于 Triton 的核函数实现。唯一的变化在于 `enqueue`。完整实现可在 `circ_pad_plugin_triton.py` 中找到。

几点备注：

- Triton 也允许 JIT 编译核函数。
- CuPy 设备数组不能直接传递给 Triton 核函数——仅接受 Torch 数组。不过，我们可以使用 `torch.as_tensor()` 来解决这一限制。
- Triton 似乎不允许指定 CUDA 流。

## Python 插件：Numba

Numba 实现可在 `circ_pad_plugin_numba.py` 中找到。几点备注：

- Numba 也允许 JIT 编译核函数。
- CuPy 设备数组可以毫无问题地传递给 Numba 核函数，因为 CuPy 数组实现了 `__cuda_array_interface__`。

## Python 插件：Torch

`enqueue()` 接口的灵活性意味着不一定总是需要实现一个自定义核函数。在这种情况下，PyTorch 的 https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html 提供了我们想要的确切功能，因此我们可以在 `enqueue()` 中使用它，如 `circ_pad_plugin_torch.py` 所示。

## Python 插件：多策略、多插件（基于 IPluginV3）

完整实现可在 `circ_pad_plugin_multi_tactic.py` 中找到。

### 自定义策略

当有多个选项可用于计算同一算子，且无法可靠预测哪种选项对于预期的输入形状/类型或目标平台更快时，让 TensorRT 在构建阶段对所有可用选项进行计时是很有用的。在 V2 插件中，TensorRT 只会计时插件支持的不同的类型/格式组合，但 V3 插件允许用户指定任意数量的自定义策略进行计时（除了类型/格式组合之外）。

在本示例中，我们指定了两个自定义策略：PyTorch 的 https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html 和使用 OpenAI Triton 编写的自定义核函数。

可以为特定的格式组合指定策略。例如，在本示例中，我们可以为 FP32 I/O 支持这两种策略，而为 FP16 I/O 仅支持 OpenAI Triton 策略。要实现这一点，请在 `get_valid_tactics()` 中返回紧接前一次调用 `configure_plugin()` 所指示的格式组合 `f` 下插件支持的策略集 `T(f)`。要在此示例中启用此行为，请传递标志 `--per-format-tactics`。

### 多个插件实例

假设您预计网络中会有多个相同插件的实例，它们将处理不同的输入，但这些输入的输入输出形状/格式以及其他决定性插件属性是相同的。使用 V2 插件时，TensorRT 会在引擎构建期间对所有此类插件实例进行计时——但这效率低下，因为这些实例之间唯一的显著区别是输入张量的值。

为了向 TensorRT 传达您希望缓存相似插件实例的计时结果，V3 插件允许指定计时缓存 ID。计时缓存 ID 应仅捕获插件 I/O 之外的计时决定因素，例如它们的形状和格式。通常，这将是插件实例之间可能不同的任何插件属性的值。

在本示例中，

- `pads` 参数的形状会影响计时，但仅限于它影响输出形状的程度。因此，计时缓存 ID 可以是空字符串。
- 我们考虑这样一种场景：有两个配置完全相同的循环填充插件实例。因此，TensorRT 应该只对其中一个实例进行计时。这可以通过检查日志来验证。

# 局限性

- 插件无法序列化到引擎中（与 `IBuilderConfig::setPluginsToSerialize()` 相反）
  - 插件类和插件创建器类必须存在于反序列化引擎所在的模块中
- 引擎 / ONNX 模型无法在 Python 之外运行（例如使用 `trtexec`）
  - 此功能可以实现，但代价是将 Python 解释器嵌入到 TRT 运行时 / 加载引擎的二进制文件中
- （仅限 `IPluginV2DynamicExt`）尚未提供非纯虚函数 `attachToContext()` 和 `detachFromContext()` 的绑定。
- `circ_pad_plugin_torch.py` 可能在 aarch64 平台上工作，但不受官方支持。

# 常见问题解答

1. 基于 Python 的插件与 C++ 插件相比，性能影响如何？

   在初步测试中，发现 Python 的开销非常小，甚至可以忽略不计。事实上，如果核函数是 AOT（而非 JIT）编译的，那么 CuPy 和 Triton 版本的插件性能与 C++ 版本相当。然而，使用 Numba 时，似乎存在显著的核函数启动开销。

2. 我能否在没有 Python 的运行时环境中部署包含 Python 插件的 TRT 引擎？

不能。无法将 Python 插件完全嵌入到引擎中，使其在推理时无需 Python 即可执行。

正是这一设计原则使得 `enqueue()` 可以用任何选择的框架来实现。

# 许可证

有关使用、复制和分发的条款和条件，请参阅 https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html 文档。

# 更新日志

2025 年 10 月：迁移至强类型 API。

2025 年 8 月：移除对 Python 3.10 以下版本的支持。

2023 年 7 月：本示例首次发布。

# 已知问题

本示例中暂无已知问题。
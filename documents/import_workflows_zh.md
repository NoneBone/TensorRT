# TensorRT 导入工作流 — 分步指南

本指南与 [`supported_models.md`](supported_models.md)一起，将先前分散在发行说明、示例、博客和论坛帖子中的 TensorRT 导入路径指导进行了集中整合。

TensorRT 支持多种将训练好的模型导入优化推理引擎的路径。本指南将逐一完整讲解每条路径——涵盖安装、导出、构建、验证——并提供可运行的命令以及行内标注的最常见陷阱。

## 目录

- [选择路径](#choosing-a-path)
- [通用前置条件](#common-prerequisites)
- [路径 1：ONNX → TensorRT](#path-1-onnx--tensorrt)
- [路径 2：Torch-TensorRT（原生 PyTorch）](#path-2-torch-tensorrt-pytorch-native)
- [路径 3：Hugging Face Hub 模型 → TensorRT](#path-3-hugging-face-hub-models--tensorrt)
- [路径 4：直接网络定义 API（C++/Python）](#path-4-direct-network-definition-api-cpython)
- [添加自定义算子 / 插件](#adding-a-custom-operator--plugin)
- [用于导出的 AI 辅助模型重写](#ai-assisted-model-rewriting-for-export)
- [验证引擎](#verifying-an-engine)
- [工具参考](#tools-reference)
- [故障排查与洞见](#troubleshooting--insights)

------

## 选择路径

| 你拥有…                                  | 推荐路径                                                     | 备注                                                         |
| ---------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 来自任何框架的 ONNX 文件                 | [ONNX → TensorRT](#path-1-onnx--tensorrt) | 最具可移植性的路径。可通过 `trtexec`、Python API 或 Polygraphy 构建。 |
| 训练好的 PyTorch 模型，希望最快上手      | [Torch-TensorRT](#path-2-torch-tensorrt-pytorch-native) | 以 Python 为主，保持在 PyTorch 生态内。最适合迭代开发。      |
| Hugging Face Hub 模型（LLM、扩散模型等） | [Hugging Face Hub 模型](#path-3-hugging-face-hub-models--tensorrt) | 大多数模型采用 导出 → ONNX → TRT；LLM 生成则直接使用 TensorRT-LLM。 |
| 使用 C++ 编写的模型架构或自定义研究栈    | [网络定义 API](#path-4-direct-network-definition-api-cpython) | 控制力最大，工作量也最大。                                   |
| 现有的 TRT 计划文件，仅需运行            | 参见 [验证引擎](#verifying-an-engine) | 本指南不涵盖此内容 —— 请参阅关于反序列化的开发者指南。       |

------

## 通用前置条件

以下所有路径均假设：

1. 支持的 NVIDIA GPU（参见 [支持矩阵](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)）。
2. 与你的 TensorRT 版本匹配的 NVIDIA 驱动 + CUDA。对于 TRT 11.x：**CUDA 13.x**。
3. 如果使用 Python API，则需要 Python 3.10+。C++ 路径需要 C++17 编译器。

### 安装 TensorRT（Python，pip）

```
# Python TRT 运行时 + Python 绑定
pip install --extra-index-url https://pypi.nvidia.com tensorrt-cu13
```

> **注意：** 对于 TRT 11.x，始终使用 `-cu13`包。不要混合使用 `-cu12`的 wheel 文件。

### 安装 TensorRT（系统包）

请遵循 [安装指南](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/overview.html)获取 `.deb`/ `.tar`/ 容器选项。NGC 容器 `nvcr.io/nvidia/tensorrt:<tag>`是获取已知良好环境的最快方式。

### 验证安装

```
python3 -c "import tensorrt; print(tensorrt.__version__)"
trtexec --help | head -5
```

------

## 路径 1：ONNX → TensorRT

ONNX 路径是将模型从 PyTorch、TensorFlow、JAX 或任何带有 ONNX 导出器的框架引入的最具可移植性的方式。

> **ONNX 兼容性说明：** TensorRT 并不支持每个 ONNX 算子或每个 ONNX opset 版本。算子和 opset 的覆盖范围取决于 TensorRT 的版本，因此请使用 `trtexec`或 Polygraphy 验证导出的模型，并准备好更新导出器/opset、重写不支持的子图，或提供自定义插件。

### 1. 导出到 ONNX

**PyTorch（dynamo 导出器，TRT 11+ 首选）：**

```
import torch

model = MyModel().eval().cuda()
example = torch.randn(1, 3, 224, 224, device="cuda")

onnx_program = torch.onnx.export(
    model,
    (example,),
    "model.onnx",
    dynamo=True,            # 使用 dynamo 导出器
    dynamic_shapes=None,    # 或使用 torch.export.Dim 指定
)
```

**TensorFlow / Keras：** 使用 `tf2onnx`：

```
python -m tf2onnx.convert --saved-model ./saved_model --output model.onnx --opset 20
```

### 2. （可选）简化与清理

```
pip install onnx onnxsim polygraphy
python -m onnxsim model.onnx model.sim.onnx
polygraphy surgeon sanitize model.sim.onnx -o model.clean.onnx --fold-constants
```

### 3. 构建 TensorRT 引擎

ONNX 文件必须先转换为序列化的 TensorRT 计划文件（`.plan`/ `.engine`），运行时才能执行它 —— TensorRT 运行时反序列化的是计划文件，它**不会**解析 ONNX。以下三个选项是可互换的前端，它们在底层调用相同的 `IBuilder`+ `nvonnxparser::IParser`；请根据你的工作流程选择。

> **关于 ONNX Runtime 的说明：** 如果你看到“TensorRT 直接执行 ONNX”，那指的是 ONNX Runtime 的 `TensorrtExecutionProvider`，它会在首次调用时内部惰性构建 TRT 引擎。这是 ORT 集成了 TRT，而非 TRT 运行时本身。

**选项 A — `trtexec`（CLI，尝试最快）：**

```
trtexec \
  --onnx=model.clean.onnx \
  --saveEngine=model.plan \
  --memPoolSize=workspace:4096 \
  --fp16                        # 或 --bf16, --int8, --fp8（取决于平台）
```

对于动态形状，添加：

```
--minShapes=input:1x3x224x224 \
  --optShapes=input:8x3x224x224 \
  --maxShapes=input:16x3x224x224
```

**选项 B — Python（`tensorrt.Builder`+ `OnnxParser`）：**

```
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
network = builder.create_network(flags)

parser = trt.OnnxParser(network, logger)
with open("model.clean.onnx", "rb") as f:
    assert parser.parse(f.read()), [parser.get_error(i) for i in range(parser.num_errors)]

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
serialized = builder.build_serialized_network(network, config)
open("model.plan", "wb").write(serialized)
```

**选项 C — Polygraphy（可编写脚本，适合流水线）：**

```
polygraphy convert model.clean.onnx \
  --convert-to trt \
  --fp16 \
  --workspace 4G \
  --trt-min-shapes input:[1,3,224,224] \
  --trt-opt-shapes input:[8,3,224,224] \
  --trt-max-shapes input:[16,3,224,224] \
  -o model.plan
```

C++ 用户：参见 `samples/sampleOnnxMNIST/`获取等效的 `IBuilder`+ `IParser`流程。

### 4. 运行引擎

参见 [验证引擎](#verifying-an-engine)。

### 常见陷阱

- **不支持的算子。** `trtexec`会指出算子名称。选项：更新你的导出器/opset、重写子图，或编写 [自定义插件](#adding-a-custom-operator--plugin)。
- **形状推断失败。** 运行 `polygraphy inspect model model.onnx --show attrs`确认每个张量都有已知的秩。
- **常量折叠意外。** `polygraphy surgeon sanitize --fold-constants`通常可以移除导出期间引入的虚假动态轴。
- **`INT64`张量。** TRT 会警告并将其转换为 `INT32`；如果值超过 `INT32`范围，请先进行清理。

------

## 路径 2：Torch-TensorRT（原生 PyTorch）

Dynamo 前端（`torch.compile(backend="tensorrt")`）是**活跃且首选**的路径。基于 JIT/追踪的 `torch_tensorrt.compile`仍然可用，但几乎不再投入新资源。

### 1. 安装

```
pip install --extra-index-url https://pypi.nvidia.com torch-tensorrt tensorrt-cu13
```

### 2. 编译（AOT — 生成独立工件）

```
import torch
import torch_tensorrt as torch_trt

model = MyModel().eval().cuda().to(torch.float16)
example = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float16)

trt_gm = torch_trt.dynamo.compile(
    torch.export.export(model, (example,)),
    inputs=[example],
    enabled_precisions={torch.float16},
    workspace_size=4 << 30,
)

# 保存并重新加载
torch_trt.save(trt_gm, "model.ep", inputs=[example])
loaded = torch.export.load("model.ep").module()
```

### 3. 编译（JIT — 首次调用触发编译）

```
import torch

compiled = torch.compile(model, backend="tensorrt", options={"enabled_precisions": {torch.float16}})
out = compiled(example)   # 首次调用触发 TRT 编译 + 缓存
```

### 4. 运行

```
with torch.no_grad():
    y = trt_gm(example)   # 或 compiled(example)
```

### 常见陷阱

- **图中断** 会回退到 eager 模式。使用 `TORCH_LOGS="graph_breaks"`进行检查。通过提升 Python 条件判断、避免 `.item()`调用以及在可能的情况下使用 `torch.cond`来消除它们。
- **动态形状** 需要为 AOT 显式添加 `torch.export.Dim(...)`注解。JIT 可以处理，但可能会针对每种形状重新编译。
- **自定义算子 / 插件。** Torch-TRT 转换器位于 Torch-TensorRT 仓库的 `core/conversion/converters/`目录下。要添加一个转换器，请参阅 [Torch-TensorRT 转换器指南](https://docs.pytorch.org/TensorRT/contributors/writing_converters.html)。

### 历史问题的洞见

- PyTorch 2.4+ 中的 `torch.export`是稳定的 Dynamo AOT 所必需的。早期版本会回退到 torchscript 追踪，而后者已被弃用。
- 混合精度：除非特定层出现精度损失，否则首选 `enabled_precisions={torch.float16}`而非 `torch.float32`。对于 BF16 目标（Blackwell、Hopper），使用 `{torch.bfloat16}`并相应转换输入。

------

## 路径 3：Hugging Face Hub 模型 → TensorRT

> **对于 LLM 生成，请直接使用 [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)。** 它是 NVIDIA 针对 Hugging Face LLM 的活跃、生产级路径 —— 处理 KV 缓存、批处理、分页注意力、FP8/INT4 量化、推测解码以及多 GPU 张量/流水线并行。下文选项 C 中描述的 `optimum-nvidia`封装器自 2026 年第二季度起已超过一年未发布，不建议用于新工作。

对于非 LLM 的 Hugging Face Hub 模型（编码器、视觉、扩散组件、语音），首选选项 A。

### 选项 A — 导出到 ONNX，然后走路径 1（推荐的默认方式）

大多数 HF 模型都能通过 `optimum`的 ONNX 导出器干净地导出：

```
pip install optimum-onnx    # ONNX 集成已在 v2 中移出 `optimum` 包
optimum-cli export onnx \
  --model google-bert/bert-base-uncased \
  --task feature-extraction \
  bert_onnx/

trtexec --onnx=bert_onnx/model.onnx --saveEngine=bert.plan --fp16
```

然后通过标准的 [路径 1 构建](#3-build-a-tensorrt-engine)运行。这是最耐用的 HF → TRT 路径，因为它仅依赖于积极维护的组件（`optimum-onnx`、`trtexec`/Python 构建器）。

### 选项 B — Torch-TensorRT

涵盖于 [路径 2](#path-2-torch-tensorrt-pytorch-native)。使用 `transformers`加载，移至 CUDA，并通过 `torch_tensorrt.dynamo.compile`或 `torch.compile(backend="tensorrt")`进行编译。当你希望留在 PyTorch 内部并快速迭代时，这是一个很好的选择。

### 选项 C — `optimum-nvidia`（便捷封装器；上游已停滞）

```
pip install optimum-nvidia
from optimum.nvidia import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    use_fp8=False,  # 或在 Hopper/Blackwell 上设为 True
)
out = model.generate(input_ids, max_new_tokens=128)
```

> **状态警告：** 最后一个 `optimum-nvidia`版本（`v0.1.0b9`）发布于 2025-01-21，此后没有任何发布。它仍然通过 `third-party/`固定了一个较旧的 `tensorrt-llm`。在采用前，请验证固定的版本是否与你的 TRT/CUDA 栈匹配，并且对于任何面向生产的场景，建议直接使用 TensorRT-LLM（参见本节顶部的提示）。

### 常见陷阱

- **分词器填充。** Hugging Face 默认为右填充；某些解码器模型期望生成时使用左填充。不匹配会导致静默错误的 logits。
- **KV 缓存形状。** 对于生成式模型，序列轴上的动态形状是强制性的。使用 `optimum-nvidia`或手动编写形状配置文件。
- **扩散流水线** 必须按组件拆分（文本编码器、UNet/DiT、VAE）—— TRT 无法摄入整个流水线。有关逐组件支持，请参阅 [supported_models.md](supported_models.md)。

------

## 路径 4：直接网络定义 API（C++/Python）

仅当 ONNX 和 PyTorch 都无法表达你的需求时才使用此路径（例如，自定义研究架构、对层选择进行极严格控制）。

```
import numpy as np
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

x = network.add_input("x", trt.float16, (-1, 3, 224, 224))
w = trt.Weights(np.random.randn(64, 3, 7, 7).astype(np.float16))
conv = network.add_convolution_nd(x, 64, (7, 7), w, trt.Weights())
conv.stride_nd = (2, 2)
network.mark_output(conv.get_output(0))

config = builder.create_builder_config()
profile = builder.create_optimization_profile()
profile.set_shape("x", (1, 3, 224, 224), (8, 3, 224, 224), (16, 3, 224, 224))
config.add_optimization_profile(profile)

plan = builder.build_serialized_network(network, config)
open("model.plan", "wb").write(plan)
```

C++ 版本遵循相同的结构；可运行的参考示例请参见 `samples/sampleINT8API`和 `samples/python/refactored/2_construct_network_with_layer_apis/`。

------

## 添加自定义算子 / 插件

当导入器报告不支持的算子时：

1. **检查 `tensorrt.IPluginRegistry`** —— 该算子可能已经有一个你尚未加载的插件。
2. **编写一个实现 `IPluginV3`的插件**（TRT 10+ 首选）。
3. **通过 `REGISTER_TENSORRT_PLUGIN`（C++）或 `trt.get_plugin_registry().register_creator(...)`（Python）注册它。**
4. **接入 ONNX** 通过在导出期间将算子命名为 `mydomain::MyPlugin`并提供匹配的插件名称。
5. **Torch-TensorRT 自定义转换器** 位于 `core/conversion/converters/`目录下 —— 请参阅 Torch-TRT 文档。

可运行的参考示例：本仓库中的 `samples/python/aliased_io_plugin/`。

有关迁移详情，请参阅 TensorRT 开发者指南中关于 [将 V2 插件迁移到 IPluginV3](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html#migrating-v2-plugins-to-ipluginv3)的部分。

------

## 用于导出的 AI 辅助模型重写

Hugging Face Hub 模型通常在第一次尝试时无法干净地导出。现代库代码使用的模式 —— 复数运算、数据依赖的控制流、非张量前向参数、变长输出 —— 是 `torch.export`/ `torch.onnx.export`/ Torch-TensorRT 无法直接追踪的。通常的解决方法**不是**在磁盘上修改上游库，而是在导出前**在运行时动态修补等效的、导出友好的变体**，然后将编译好的模块透明地插回流水线。

这是一项重复性的、需要精细作用域控制的工作：阅读上游实现，识别破坏导出器的单一模式，编写行为等效的替代品，同时保留库其余部分所期望的一切。这**正是 AI 编码代理能够发挥作用的那类任务** —— 代理读取数百行上游源代码（diffusers、transformers），提出等效的公式，并针对追踪器错误进行迭代，而不会丢失上下文。下面的实际示例是针对一个非平凡扩散流水线的此过程的产出。

### 实际示例：Qwen-Image（`diffusers`）→ Torch-TensorRT AOT

以下精简模式源自一个 Qwen-Image Torch-TensorRT AOT 验证脚本，旨在为开源用户提供自包含的代码。

该脚本通过 `torch_tensorrt.dynamo.compile`编译 `QwenImagePipeline`的所有三个重量级组件 —— 文本编码器、MMDiT Transformer、VAE 解码器 —— 并将编译好的模块重新注入流水线。必须修复五个不同的导出阻碍点，每一个都代表了更广泛的类别：

#### 1. 复数 RoPE 数学 → 预计算实数值 cos/sin

Diffusers 的 `QwenEmbedRope`将旋转嵌入频率存储为 `torch.complex64`缓冲区，并在前向路径中调用 `torch.view_as_real(...)`。Torch-TensorRT 的复数图检测无法处理残留的复数算子并导致段错误。

**修复：** 在模块上预生成实数值的 `cos`/`sin`缓冲区，修补前向路径使其从中读取，**但保留原始的复数缓冲区不变**（导出器在入口处仍会探测它们）：

```
pos = torch.view_as_real(module.pos_freqs)
module._real_pos_cos = pos[..., 0].repeat_interleave(2, dim=-1).contiguous()
module._real_pos_sin = pos[..., 1].repeat_interleave(2, dim=-1).contiguous()
# 不要覆盖 pos_freqs/neg_freqs —— complex_graph_detection 仍然会读取它们。
```

代理在此处的作用：阅读 `diffusers/models/transformers/transformer_qwenimage.py`（约 1500 行），精确定位触及复数张量的两个 `forward`方法，并在不改变数值的情况下推导出实数值等效方案。

#### 2. 非张量前向参数 → 将它们烘焙进包装器中

Transformer 接收 `img_shapes: list[list[tuple[int, int, int]]]`，这是 `torch.export`拒绝追踪的。

**修复：** 一个轻量级包装器将该形状列表存储为构造函数参数，以便导出的前向签名仅为纯张量：

```
class QwenImageTransformerAOTWrapper(nn.Module):
    def __init__(self, transformer, img_shapes):
        super().__init__()
        self.transformer = transformer
        self.img_shapes = img_shapes
    def forward(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, timestep):
        return self.transformer(
            hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask, timestep=timestep,
            img_shapes=self.img_shapes, return_dict=False)[0]
```

#### 3. HF 风格的输出数据类 → 导出前解包，返回时重新包装

流水线期望类似 `Transformer2DModelOutput(sample=...)`或带有 `.hidden_states`的对象作为输出。Torch-TRT 需要纯张量。通过两层解决：

- **导出包装器**（`TextEncoderAOTWrapper`、`VaeDecoderAOTWrapper`）返回一个裸张量。
- **重注入代理**（`CompiledTextEncoderProxy`、`CompiledTransformerProxy`、`CompiledVAEProxy`）暴露下游代码在原模块上读取的每个属性（`config`、`dtype`、`device`、`cache_context()`等），并将输出重新包装成预期的数据类，以便周围的流水线代码察觉不到替换。

#### 4. 变长分词 → 强制静态提示形状

默认的 `encode_prompt`路径会根据注意力掩码对每个样本进行切片，产生变长的隐藏状态。Torch-TRT 要求 AOT 具有静态形状。

**修复：** 将 `_get_qwen_prompt_embeds`动态修补到流水线上，以便分词始终产生与 TRT 文本编码器编译时相同的 `[B, S]`形状：

```
pipe._get_qwen_prompt_embeds = MethodType(_get_qwen_prompt_embeds_fixed, pipe)
```

代理在此处的作用：定位流水线分发到的（未记录的）方法，用固定的 `max_seq_len`复现修剪/填充逻辑，并确保与编译模块的数据类型/设备对齐。

#### 5. 内存受限的编译 → 提示资源分区器

如果贪婪地编译所有内容，完整的流水线会超出单个 GPU 的工作集：

```
import torch_tensorrt as torch_trt

# cpu_memory_budget 以字节为单位 —— 根据你的主机内存余量进行调整。
CPU_MEMORY_BUDGET_BYTES = 32 * 1024**3  # 32 GiB

torch_trt.compile(module, ir="dynamo", arg_inputs=inputs,
                  require_full_compilation=False,
                  enable_resource_partitioning=True,
                  cpu_memory_budget=CPU_MEMORY_BUDGET_BYTES,
                  truncate_double=True, optimization_level=1)
```

### 模式要点

将此方案应用于任何非平凡的 HF 导出：

1. **先运行朴素导出。** 让追踪器/导出器失败并仔细阅读错误 —— 失败的算子/模式会告诉你修补什么。
2. **在运行时修补，而不是在磁盘上修补。** 从你的导出脚本中动态修补上游模块，这样你就永远不需要 Fork 该库。
3. **为导出器包装；为流水线代理。** `*AOTWrapper`解包 HF 输出以供导出；`Compiled*Proxy`在返回时重新包装它们，并携带下游代码读取的每个属性。
4. **保留未被观察到的不变性。** 当导出器探测一个缓冲区时（例如，`complex_graph_detection`读取 `pos_freqs`），不要覆盖该缓冲区 —— 添加一个并行的实数值缓冲区。
5. **在循环中通过代理进行迭代。** 上述每个修复都需要针对上游源代码进行一到两次读取-诊断-修补循环；代理可以比人类浏览陌生库代码更快地执行这些循环，而你只需审查差异。

Qwen-Image 的最终结果：一个流水线，其重量级组件全部在 Torch-TensorRT 上运行，对安装的 `diffusers`/ `transformers`/ `torch_tensorrt`零改动，并且在固定随机种子下生成的图像与 eager 流水线输出相匹配。

------

## 验证引擎

```
# 健全性检查性能和数值
trtexec --loadEngine=model.plan --shapes=input:1x3x224x224 --verbose

# 针对 ONNX 源进行并排精度对比
polygraphy run model.onnx --trt --onnxrt \
  --atol 1e-3 --rtol 1e-3 --input-shapes input:[1,3,224,224]
```

对于 LLM 风格的生成，在信任新引擎之前，请在确定性随机种子下与参考实现进行逐 token 比较。

------

## 工具参考

| 工具                | 功能                                           | 安装                            |
| ------------------- | ---------------------------------------------- | ------------------------------- |
| `trtexec`           | 从 ONNX 或序列化计划文件构建 + 运行 + 分析引擎 | 随 TRT 捆绑                     |
| `polygraphy`        | 在每个阶段检查、清理、比较和调试模型           | `pip install polygraphy`        |
| `onnxsim`           | 折叠常数并简化 ONNX 图                         | `pip install onnxsim`           |
| `onnx-graphsurgeon` | 程序化 ONNX 图编辑                             | `pip install onnx-graphsurgeon` |
| `nsys`/ `ncu`       | 运行时分析和内核分析                           | NVIDIA CUDA Toolkit             |

------

## 故障排查与洞见

来自客户报告问题的共享经验。请自由扩展此列表 —— 这是本指南中唯一最有价值的部分。

- **“引擎计划文件在不兼容的设备上生成”** —— 计划文件不能跨计算能力移植。请在部署 GPU 上重新构建，或在构建时针对多个 SM。
- **与框架的精度差距** —— 从 `polygraphy run ... --onnxrt --trt --atol ...`开始定位。如果 FP16 引擎出现偏差，尝试对受影响的子图使用 `--stronglyTyped`+ 显式 FP32 转换。
- **构建期间内存不足** —— 降低 `--memPoolSize=workspace:N`或禁用你不需要的策略来源（`--tacticSources=-CUBLAS_LT`）。
- **首次推理缓慢** —— CUDA 内核 JIT + 计划文件反序列化成本是一次性的。在计时前进行 ≥3 次预热迭代。
- **`IShapeLayer`/ 数据依赖的形状** —— 某些模式（例如，带有动态输出形状的 `where(cond, x, y)`）需要 `IShapeLayer`+ 配置文件形状张量。请参阅开发者指南中关于动态形状的章节。
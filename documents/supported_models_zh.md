# TensorRT 支持的模型列表

本已验证模型矩阵与 ./import_workflows.md 相对应。针对每个模型系列，列出了验证过程中使用的 dtype（数据类型）。

## 范围与阅读指南

TensorRT 是一个通用的神经网络图执行引擎，而非模型库。原则上，**任何神经网络架构**只要能通过./import_workflows.md中描述的工作流进行表达，都可以在 TensorRT 上运行。./import_workflows.md#adding-a-custom-operator--plugin章节涵盖了针对 TensorRT 尚未原生实现的算子的兜底方案。

下表**并非**详尽的支援列表。它是 NVIDIA 已验证和基准测试的模型子集；我们发布此表是为了让你了解哪些配置拥有已知良好的基线，以及当前的粗糙边缘在哪里。如果你的模型未列出，我们仍然期望它能够正常工作——如果不行，请提交 Issue。

### 阅读表格

- **Dtype（数据类型）** 列出了验证基线所使用的精度。其他精度也可能适用。
- 组件拆分的模型（扩散管道、带编码器/解码器的语音模型）会为每个已验证的组件单独列出一行。

## 目录

- #llms--文本生成
- #encoder-only-nlp-bert-family-embeddings
- #vision-classification--embeddings
- #speech--audio
- #diffusion-models
- #multimodal
- #legacy--trt-sample-models
- #requesting-new-model-coverage

---

## LLMs / 文本生成

> **LLM 生成的首选路径：** https://github.com/NVIDIA/TensorRT-LLM（支持 KV 缓存、Paged Attention、FP8/INT4、推测解码、张量/流水线并行）。对于生产级 LLM 服务，请使用 TensorRT-LLM。

| 模型                                   | Dtype    |
|----------------------------------------|----------|
| `meta-llama/Llama-3.1-8B`              | bfloat16 |
| `meta-llama/Llama-3.2-1B`              | bfloat16 |
| `Qwen/Qwen3-0.6B`                      | bfloat16 |
| `deepseek-ai/Janus-Pro-7B`             | bfloat16 |

> 有关 TensorRT-LLM 自身的覆盖范围，请参阅 https://github.com/NVIDIA/TensorRT-LLM#model-zoo。

---

## 仅编码器 NLP（BERT 系列、嵌入模型）

| 模型                                                   | Dtype   |
|--------------------------------------------------------|---------|
| `google-bert/bert-base-uncased`                        | float32 |
| `google-bert/bert-base-multilingual-cased`             | float16 |
| `FacebookAI/roberta-base`                              | float32 |
| `FacebookAI/roberta-large`                             | float32 |
| `FacebookAI/xlm-roberta-base`                          | float32 |
| `distilbert/distilbert-base-uncased`                   | float32 |
| `sentence-transformers/all-MiniLM-L6-v2`               | float32 |
| `sentence-transformers/all-mpnet-base-v2`              | float32 |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | float32 |
| `BAAI/bge-base-en-v1.5`                                | float32 |
| `nlpaueb/legal-bert-base-uncased`                      | float32 |

---

## 视觉分类与嵌入

| 模型                                         | Dtype   |
|----------------------------------------------|---------|
| `torchvision/resnet50`                       | float32 |
| `timm/mobilenetv3_small_100.lamb_in1k`       | float32 |
| `trpakov/vit-face-expression`                | float32 |
| `openai/clip-vit-base-patch32`               | float32 |
| `openai/clip-vit-large-patch14`              | float32 |
| `facebook/dinov2-base`                       | float32 |
| `Falconsai/nsfw_image_detection`             | float32 |
| `dima806/fairface_age_image_detection`       | float32 |

---

## 语音 / 音频

| 模型（组件）                                      | Dtype   |
|---------------------------------------------------|---------|
| `openai/whisper-large-v3-turbo` (Encoder)         | float32 |
| `openai/whisper-large-v3-turbo` (Decoder)         | float32 |
| `openai/whisper-large-v3` (Encoder)               | float32 |
| `openai/whisper-large-v3` (Decoder)               | float32 |
| `laion/clap-htsat-fused`                          | float32 |
| `sesame/csm-1b` (Backbone)                        | float32 |
| `neuphonic/neutts-air`                            | float32 |
| `LiquidAI/LFM2-Audio-1.5B`                        | float32 |

---

## 扩散模型

扩散管道按组件（文本编码器 / UNet 或 DiT / VAE）进行评估，因为 TRT 不直接摄取管道对象。

| 管道（组件）                                                  | Dtype    |
|---------------------------------------------------------------|----------|
| `stabilityai/sd-turbo`                                        | float16  |
| `stabilityai/sdxl-turbo` (UNet)                               | float16  |
| `stabilityai/sdxl-turbo` (VAE / Text Encoders)                | mixed    |
| `stabilityai/stable-diffusion-xl-base-1.0`                   | float16  |
| `CompVis/stable-diffusion-v1-4`                               | float16  |
| `stable-diffusion-v1-5/stable-diffusion-v1-5`                | float16  |
| `stabilityai/stable-diffusion-2-1`                            | float16  |
| `playgroundai/playground-v2.5-1024px-aesthetic`               | float16  |
| `dataautogpt3/ProteusV0.3`                                    | float16  |
| `black-forest-labs/FLUX.2-dev` (Text Encoder)                 | bfloat16 |
| `black-forest-labs/FLUX.2-dev` (DiT)                          | bfloat16 |
| `black-forest-labs/FLUX.2-dev` (VAE)                          | float16  |
| `black-forest-labs/FLUX.1-schnell` (DiT / TextEnc / VAE)      | mixed    |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (Text Encoder)             | float16  |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (VAE)                      | float16  |
| `Qwen/Qwen-Image` (Text Encoder)                              | bfloat16 |
| `Qwen/Qwen-Image` (DiT / VAE)                                 | bfloat16 |
| `stabilityai/stable-diffusion-3-medium-diffusers`             | bfloat16 |
| `stabilityai/stable-diffusion-3.5-medium` / `3.5-large`       | mixed    |
| `HiDream-ai/HiDream-I1-Full`                                  | bfloat16 |
| `stabilityai/stable-video-diffusion-img2vid-xt`               | float16  |

---

## 多模态

| 模型                                  | Dtype   |
|---------------------------------------|---------|
| `openai/clip-vit-base-patch32`        | float32 |
| `deepseek-ai/Janus-Pro-7B`             | bfloat16 |
| `Datadog/Toto-Open-Base-1.0`           | float32 |

---

## 旧版 / TRT 示例模型

TensorRT 为这些经典架构和工作流提供了经过人工验证的 C++/Python 示例：

- MNIST 数字分类器、模型解析、动态形状、插件和安全运行时示例——请参阅本仓库中的 `samples/` 目录。

---

## 申请新的模型覆盖

提交 GitHub Issue 时请包含：

1.  Hugging Face ID 或模型来源 URL。
2.  目标 dtype（fp32 / fp16 / bf16 / fp8 / int8 / int4）。
3.  任何框架级别的可运行示例（有助于我们快速复现）。

维护者将对模型进行基准测试并扩展此表——基准测试步骤无需外部贡献者进行操作。
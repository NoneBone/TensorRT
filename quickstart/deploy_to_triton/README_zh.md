# TensorRT 转 Triton

en [English](./README.md)｜ zh_CN [简体中文](./README_zh.md)

本快速入门指南展示了如何在 Triton 推理服务器上部署一个经 TensorRT 加速的简单 ResNet 模型。在讨论机器学习基础设施时，优化与部署密不可分。对于 TensorRT 用户而言，通过网络级优化以获取最佳性能已是专业领域。

然而，部署此优化后的模型有其自身的考量与挑战，例如构建支持并发模型执行的基础设施，以及通过 HTTP、gRPC 等方式支持客户端请求。

[Triton 推理服务器](https://github.com/triton-inference-server/server)解决了上述问题及其他更多挑战。接下来，我们将逐步讲解使用 Torch-TensorRT 优化模型、在 Triton 推理服务器上部署模型，以及构建客户端查询模型的完整流程。

## 步骤 1：使用 TensorRT 优化模型

如果您不熟悉 TensorRT，请参考此 [视频](https://youtu.be/rK-jxPPY9V4)。本流程的第一步是使用 TensorRT 加速您的模型。为便于演示，我们假设您已拥有训练好的 ONNX 格式模型。

（可选）如果您手边没有现成的 ONNX 模型，仅想跟随教程操作，可以使用以下脚本：

```sh
# <xx.xx> 是 NVIDIA TensorRT 容器发布标签的 yy:mm 版本号；例如 24.07
# 请访问 https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch 查看最新版本

docker run -it --gpus all -v /path/to/this/folder:/resnet50_eg nvcr.io/nvidia/pytorch:<xx.xx>-py3

python export_resnet_to_onnx.py
exit
```

您可能需要在此处 [创建账户并获取 API 密钥](https://ngc.nvidia.com/setup/)。注册后使用您的密钥登录（注册后请遵循 [此处说明](https://ngc.nvidia.com/setup/api-key)）。

现在我们已经有了 ONNX 模型，便可使用 TensorRT 进行优化。这些优化结果将存储为 TensorRT 引擎，也称为 TensorRT 计划文件。

虽然安装 TensorRT 有多种方式，但最简单的方法是直接获取我们预构建的 Docker 容器。

```sh
docker run -it --gpus all -v /path/to/this/folder:/trt_optimize nvcr.io/nvidia/tensorrt:<xx:yy>-py3
```

构建 TensorRT 引擎有多种方法；在本演示中，我们将直接使用 `trtexec`[命令行工具](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec)。

```sh
trtexec --onnx=resnet50.onnx \
        --saveEngine=model.plan \
        --useCudaGraph
```

在进入下一步之前，重要的是我们需要知道网络“输入”和“输出”层的名称，因为 Triton 需要这些信息。一个简便的方法是使用 TensorRT 容器中自带的 `polygraphy`。如果您想深入了解 Polygraphy 及其用法，请访问 [此仓库](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy)。您可以查看大量 [示例](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/inspect)，了解如何使用 Polygraphy 检查模型。

```sh
polygraphy inspect model model.plan
```

输出的部分内容如下所示：

```sh
[I] ==== TensorRT Engine ====
    Name: Unnamed Network 0 | Explicit Batch Engine
    
    ---- 1 Engine Input(s) ----
    {input [dtype=float32, shape=(1, 3, 224, 224)]}
    
    ---- 1 Engine Output(s) ----
    {output [dtype=float32, shape=(1, 1000)]}
    
    ---- Memory ----
    Device Memory: 8228864 bytes
    
    ---- 1 Profile(s) (2 Tensor(s) Each) ----
    - Profile: 0
        Tensor: input           (Input), Index: 0 | Shapes: min=(1, 3, 224, 224), opt=(1, 3, 224, 224), max=(1, 3, 224, 224)
        Tensor: output         (Output), Index: 1 | Shape: (1, 1000)
    
    ---- 87 Layer(s) ----
```

至此，我们已准备好进入下一步——设置 Triton 推理服务器。

## 步骤 2：设置 Triton 推理服务器

如果您是 Triton 推理服务器的新手并希望了解更多，我们强烈推荐查看我们的 [GitHub 仓库](https://github.com/triton-inference-server)。

要使用 Triton，我们需要创建一个模型仓库。顾名思义，模型仓库是推理服务器托管模型的存储库。虽然 Triton 可以从多个仓库提供服务，但在本例中，我们将介绍最简单的模型仓库形式。使用 Triton 需要创建模型仓库，其结构应如下所示：

```sh
model_repository
|
+-- resnet50
    |
    +-- config.pbxt
    +-- 1
        |
        +-- model.plan
```

Triton 需要提供两个文件来服务模型：模型本身和一个通常以 `config.pbtxt`形式提供的模型配置文件。我们提供了一个 `config.pbtxt`示例，您可以在此特定示例中使用。该文件用于详细描述模型配置，包括输入和输出层的名称与形状、数据类型、调度与批处理细节等。如果您是 Triton 新手，我们强烈建议您查阅文档的 [此章节](https://github.com/triton-inference-server/server/blob/main/docs/getting_started/quickstart.md)以了解更多信息。

完成模型仓库设置后，即可启动 Triton 服务器。您可以使用以下 Docker 命令进行操作。

```sh
# 请确保 Triton 容器中的 TensorRT 版本
# 与用于优化模型的环境中的 TensorRT 版本一致。
# 在本示例中，<xx.yy> 使用 24.07 即可正常工作

docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models
```

## 步骤 3：使用 Triton 客户端查询服务器

在开始之前，请确保手头有一张示例图片。如果没有，请下载一张示例图片以测试推理。本节将介绍一个非常基础的客户端。如需更多完整示例，请参阅 [Triton 客户端仓库](https://github.com/triton-inference-server/client/tree/main/src/python/examples)。

```sh
wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
```

安装依赖项。

```sh
pip install torchvision
pip install attrdict
pip install nvidia-pyindex
pip install tritonclient[all]
```

构建客户端主要涉及三个基本要点：

- 首先，建立与 Triton 推理服务器的连接。

- 其次，指定模型输入和输出层的名称。

- 最后，向 Triton 推理服务器发送推理请求。

您可以在示例客户端中找到对应的函数实现。

```sh
python3 triton_client.py
```

输出结果应如下所示：

```sh
[b'12.477911:90' b'11.527293:92' b'9.662313:14' b'8.411008:136' b'8.220069:11']
```

此处的输出格式为 `<置信度分数>:<分类索引>`。要了解如何将这些映射到标签名称等信息，请参阅我们的 [文档](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md)。
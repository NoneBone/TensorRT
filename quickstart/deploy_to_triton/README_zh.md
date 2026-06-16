# TensorRT 转 Triton

本快速入门指南展示了如何在 Triton 推理服务器上部署由 TensorRT 加速的简单 ResNet 模型。在讨论机器学习基础设施时，优化和部署往往是相辅相成的。对于 TensorRT 用户而言，通过网络级优化获得最佳性能已经是专业领域的一部分。

然而，部署这些优化后的模型有其自身的考量与挑战，例如构建支持并发模型执行的基础设施、支持 HTTP 和 gRPC 等客户端协议。

https://github.com/triton-inference-server/server 解决了上述问题及更多挑战。让我们逐步探讨使用 Torch-TensorRT 优化模型、将其部署在 Triton 推理服务器上，并构建一个查询模型的客户端的过程。

## 步骤 1：使用 TensorRT 优化您的模型

如果您不熟悉 TensorRT，请参考此 https://youtu.be/rK-jxPPY9V4。该流程的第一步是使用 TensorRT 加速您的模型。为了演示目的，我们假设您已经拥有训练好的 ONNX 格式模型。

（可选）如果您手头没有 ONNX 模型只是想跟着操作，可以随时使用此脚本：
```
# <xx.xx> 是 NVIDIA TensorRT 容器发布标签的 yy:mm 版本；例如 24.07
# 请查看 https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch 获取最新版本

docker run -it --gpus all -v /path/to/this/folder:/resnet50_eg nvcr.io/nvidia/pytorch:<xx.xx>-py3

python export_resnet_to_onnx.py
exit
```

您可能需要在此处 https://ngc.nvidia.com/setup/ 创建一个账户并获取 API 密钥。注册并使用您的密钥登录（注册后请遵循此处 https://ngc.nvidia.com/setup/api-key 的说明）。

现在我们有了 ONNX 模型，可以使用 TensorRT 来优化您的模型。这些优化以 TensorRT 引擎（也称为 TensorRT plan 文件）的形式存储。

虽然安装 TensorRT 有多种方法，但最简单的方法是直接获取我们预构建的 Docker 容器。

```
docker run -it --gpus all -v /path/to/this/folder:/trt_optimize nvcr.io/nvidia/tensorrt:<xx:yy>-py3
```

构建 TensorRT 引擎有多种方式；在本演示中，我们将简单地使用 `trtexec` https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec。

```
trtexec --onnx=resnet50.onnx \
        --saveEngine=model.plan \
        --useCudaGraph
```

在进入下一步之前，了解您网络的“输入”和“输出”层的名称非常重要，因为 Triton 需要这些信息。一个简单的方法是使用 TensorRT 容器中自带的 `polygraphy`。如果您想了解更多关于 Polygraphy 及其用法，请访问此 https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy。您可以查看大量 https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/inspect，展示 Polygraphy 检查模型的实用性。

```
polygraphy inspect model model.plan
```
输出的部分内容如下所示：
```
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

至此，我们已准备好进入下一步：设置 Triton 推理服务器。

## 步骤 2：设置 Triton 推理服务器

如果您是 Triton 推理服务器的新手并想了解更多信息，我们强烈建议您查看我们的 https://github.com/triton-inference-server。

要使用 Triton，我们需要创建一个模型仓库。顾名思义，模型仓库是推理服务器托管的模型的存储库。虽然 Triton 可以从多个仓库提供服务，但在本例中，我们将讨论最简单的模型仓库形式。要使用 Triton，我们需要创建一个模型仓库。仓库的结构应如下所示：
```
model_repository
|
+-- resnet50
    |
    +-- config.pbxt
    +-- 1
        |
        +-- model.plan
```

Triton 需要提供两个文件来服务模型：模型本身和一个模型配置文件，通常以 `config.pbtxt` 的形式提供。我们提供了一个 `config.pbtxt` 的示例，您可以在此特定示例中使用。该文件用于描述确切的模型配置，包含输入和输出层的名称和形状、数据类型、调度和批处理详细信息等。如果您是 Triton 的新手，我们强烈建议您查看文档的 https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md 以了解更多信息。

一旦设置好模型仓库，就可以启动 Triton 服务器了。您可以使用下面的 Docker 命令来完成。
```
# 请确保 Triton 容器中的 TensorRT 版本
# 与用于优化模型的环境中的 TensorRT 版本一致
# 在本示例中，<xx.yy> 使用 24.07 即可正常工作


docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models
```

## 步骤 3：使用 Triton 客户端查询服务器

在继续之前，请确保手头有一张示例图片。如果没有，请下载一张示例图片以测试推理。在本节中，我们将介绍一个非常基础的客户端。如需更多详尽的示例，请参阅 https://github.com/triton-inference-server/client/tree/main/src/python/examples。

```
wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
```

安装依赖项。
```
pip install torchvision
pip install attrdict
pip install nvidia-pyindex
pip install tritonclient[all]
```

构建客户端主要涉及三个基本要点。
* 首先，我们与 Triton 推理服务器建立连接。
* 其次，我们指定模型的输入和输出层名称。
* 最后，我们向 Triton 推理服务器发送推理请求。

您可以在示例客户端中找到对应的函数。
```
python3 triton_client.py
```
其输出应如下所示：
```
[b'12.477911:90' b'11.527293:92' b'9.662313:14' b'8.411008:136' b'8.220069:11']
```
这里的输出格式为 `<置信度分数>:<分类索引>`。要了解如何将这些映射到标签名称等信息，请参阅我们的 https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md。
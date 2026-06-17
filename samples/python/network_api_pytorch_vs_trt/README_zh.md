# 使用 PyTorch 和 Python 的 TensorRT “Hello World” 示例

**目录**

- #description
- #how-does-this-sample-work
    * #tensorrt-api-layers-and-ops
- #prerequisites
- #running-the-sample
    * #sample-help-options
- #additional-resources
- #license
- #changelog
- #known-issues

## 描述

本示例 `network_api_pytorch_mnist` 在 https://ossci-datasets.s3.amazonaws.com/mnist/ 数据集上训练一个卷积模型，并使用 TensorRT 引擎运行推理。

1. 构建 py 对比 trt 的推理。
2. 全局的显存统计，而非简单的 torch。
2. 支持动态批尺寸设置，观察耗时与显存变化。
3. 实现 stream overlap。

## 示例工作原理

本示例是一个端到端示例，涵盖在 PyTorch 中训练模型、在 TensorRT 中重建网络、导入训练好的模型权重，并最终使用 TensorRT 引擎运行推理。更多信息，请参阅 https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#network_python。

`sample.py` 脚本从 `mnist.py` 脚本中导入函数，用于训练 PyTorch 模型以及从 PyTorch 数据加载器中获取测试用例。

### TensorRT API 层和操作

本示例使用以下层。有关这些层的更多信息，请参阅 https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers 文档。

https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer
激活层实现逐元素的激活函数。具体而言，本示例使用类型为 `RELU` 的激活层。

https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#convolution-layer
卷积层计算 2D（通道、高度和宽度）卷积，可带偏置或不带偏置。

https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#matrixmultiply-layer
矩阵乘法层实现矩阵乘法运算。
（https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#fullyconnected-layer 自 8.4 版本起已弃用。
全连接语义中的偏置可以通过 `SUM` 操作的 https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#elementwise-layer 添加。）

https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#pooling-layer
池化层在单个通道内实现池化。支持的池化类型包括 `最大值`、`平均值` 和 `最大-平均混合`。

## 先决条件

1. 升级 pip 版本并安装示例依赖项。
    ```bash
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
    ```

运行本示例必须使用 Python 3.6 或更高版本。

在 PowerPC 系统上，您需要手动使用 IBM 的 https://www.ibm.com/support/knowledgecenter/SS5SF7_1.6.0/navigation/pai_install.htm 安装 PyTorch。

2. 准备示例数据

请参阅主示例 README 中的 ../../README.md#preparing-sample-data。
    
MNIST 数据集位于 `$TRT_DATADIR/mnist` 目录下。使用： export TRT_DATADIR=/root/cys/PROJECT/00-COMMON/DEMO/02-TensorRT/data

## 运行示例

1.  运行示例以创建 TensorRT 推理引擎并运行推理：
    `python3 sample.py`

2.  验证示例是否成功运行。如果示例成功运行，您应该会看到测试用例与预测结果匹配。
     ```
    测试用例：0
    预测结果：0
     ```

### 示例 --help 选项

要查看可用选项的完整列表及其说明，请使用 `-h` 或 `--help` 命令行选项。

# 附加资源

以下资源有助于更深入地了解如何使用 Python 开始使用 TensorRT：

**模型**
- https://github.com/pytorch/examples/tree/master/mnist

**数据集**
- https://ossci-datasets.s3.amazonaws.com/mnist/

**文档**
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html

# 许可证

有关使用、复制和分发的条款和条件，请参阅 https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html 文档。

# 更新日志
2025 年 10 月
迁移至强类型 API。

2025 年 8 月
移除对低于 3.10 的 Python 版本的支持。

2023 年 8 月
移除对低于 3.8 的 Python 版本的支持。

2021 年 9 月
更新示例以使用显式批量网络定义。

2021 年 3 月
记录了 Python 版本限制。

2019 年 2 月
重新创建、更新并审核了此 `README.md` 文件。

# 已知问题

由于 `torch` 和 `torchvision` 的版本要求，本示例仅支持 Python 3.6+。
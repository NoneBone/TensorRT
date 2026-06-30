# Detectron 2 Mask R-CNN R50-FPN 3x 在 TensorRT 中的支持

 en [English](./README.md) ｜ zh_CN [简体中文](./README_zh.md)

本仓库支持 Detectron 2 Mask R-CNN R50-FPN 3x 模型在 TensorRT 中的部署。本脚本集用于辅助该模型的转换、运行与验证工作。

## 更新日志

- 2025 年 10 月

  -   迁移至强类型 API。

- 2025 年 8 月

  -   移除了对 Python 3.10 以下版本的支持。

- 2023 年 8 月

  -   更新 ONNX 版本支持至 1.14.0。

  -   更新 Python>=3.8 环境下的 ONNX Runtime 版本支持至 1.15.1。

  -   移除了对 Python 3.8 以下版本的支持。

- 2023 年 7 月：

  -   更新基准测试数据并注明所用硬件。

- 2022 年 10 月：

  -   更新转换器以支持 `tracing`导出方式，取代已弃用的 `caffe2_tracing`。

## 环境配置

为确保脚本正常运行，建议使用 TensorRT >= 8.4.1 的环境。

请按照 [TensorRT 安装指南](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)安装 TensorRT。请确保同时正确安装了 TensorRT 的 Python 绑定，可通过安装 TensorRT 下载包中的 `python3-libnvinfer`和 `python3-libnvinfer-dev`软件包来获取。

安装 `requirements.txt`中列出的所有依赖项：

```
pip install -r requirements.txt
```

注意：本示例无法在 Jetson 平台上运行，因为 `torch.distributed`不可用。若要检查平台是否支持 `torch.distributed`，可打开 Python 交互界面并确认 `torch.distributed.is_available()`返回 `True`。

## 模型转换

将 Detectron 2 Mask R-CNN R50-FPN 3x 模型转换为 TensorRT 的工作流大致为 Detectron 2 → ONNX → TensorRT，因此该过程的某些环节需要安装 Detectron 2。官方的 ONNX 导出文档详见 [此处](https://detectron2.readthedocs.io/en/latest/tutorials/deployment.html)。

‘‘‘

# 安装 DT2
git clone git@github.com:facebookresearch/detectron2.git
<!-- 关闭 build isolation，告诉 pip：别给我建临时环境，直接用我当前 cp312 里的 torch -->
python -m pip install -e ./detectron2 --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
‘‘‘

### Detectron 2 模型导出

模型导出通过 Detectron 2 [GitHub 仓库](https://github.com/facebookresearch/detectron2)中的 `detectron2/tools/deploy/export_model.py`脚本完成。Detectron 2 Mask R-CNN R50-FPN 3x 模型支持动态输入，测试阶段的最小尺寸为 800，最大为 1333。由于用于转换的 TensorRT 插件不支持动态形状，因此我们必须将输入张量的高度和宽度均固定为 1344。之所以选择 1344 而非 1333，是因为模型要求输入张量的高度和宽度必须能被 32 整除。为了以正确的 1344x1344 分辨率导出模型，我们需要对 `export_model.py`进行修改。当前第 160-162 行的代码：

```
aug = T.ResizeShortestEdge(
    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
)
```

需修改为：

```
aug = T.ResizeShortestEdge(
    [1344, 1344], 1344
)
```

⚠️ 另外在onnx.export 的入口参数中，新版 exporter 需要增添：dynamo=False 

导出脚本接受 `--sample-image`作为参数。该图像用于调整输入维度以及网络中其余张量的维度。该样本图像必须是 1344x1344 尺寸，且至少包含一个模型可检测到的物体。建议将 COCO 数据集中的某张图像上采样至 1344x1344。示例命令如下：

```
python detectron2/tools/deploy/export_model.py \
    --sample-image 1344x1344.jpg \
    --config-file detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --export-method tracing \
    --format onnx \
    --output ./model/ \
    MODEL.WEIGHTS /root/.torch/iopath_cache/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl \
    MODEL.DEVICE cuda
```

其中 `--sample-image`为 1344x1344 的图像；`--config-file`为 Detectron 2 自带的 Mask R-CNN R50-FPN 3x 配置文件路径；`MODEL.WEIGHTS`为 Mask R-CNN R50-FPN 3x 的权重文件，可从 [此处](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)下载。生成的 `model.onnx`将作为转换脚本的输入。

### 构建 ONNX 计算图

以下是受支持的 Detectron 2 模型：

| **模型**              | **分辨率** |
| --------------------- | ---------- |
| Mask R-CNN R50-FPN 3x | 1344x1344  |

若 Detectron 2 Mask R-CNN 模型已准备好进行转换（即已运行 `detectron2/tools/deploy/export_model.py`），请执行：

```
python create_onnx.py \
    --exported_onnx ./model/model.onnx \
    --onnx ./model/converted.onnx \
    --det2_config ./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --det2_weights /root/.torch/iopath_cache/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl \
    --sample_image 1344x1344.jpg
```

这将生成 `converted.onnx`文件，该文件已准备好转换为 TensorRT 引擎。

需要特别说明的是，此处的 `--sample_image`用于锚框生成。Detectron 2 的 ONNX 模型计算图中不包含锚框数据，因此必须在“离线”状态下生成锚框。若使用自定义模型，请确保模型的预处理逻辑与 `get_anchors(self, sample_image)`函数中的代码逻辑一致。

该脚本包含若干可选参数，包括：

- `--first_nms_threshold [...]`：允许覆盖默认的第一次 NMS 分数阈值参数，因为 NMS 插件的运行时延迟对该值较为敏感。在保证满足应用需求的前提下，建议将该值设得尽可能高，以降低推理延迟。在 Mask R-CNN 中，这将是区域提议网络（RPN）的分数阈值。

- `--second_nms_threshold [...]`：允许覆盖默认的第二次 NMS 分数阈值参数，可进一步优化 NMS 插件的运行时延迟。在保证满足应用需求的前提下，建议将该值设得尽可能高，以降低推理延迟。这将是第二次也是最后一次 NMS 操作。

- `--batch_size`：允许选择不同的批处理大小，默认为 1。

此外，你也可以使用 [Netron](https://netron.app/)等工具可视化生成的 ONNX 计算图。

计算图的输入是一个 `float32`张量，具有选定的输入形状，包含范围为 0 到 255 的 RGB 像素数据。所有预处理操作将在模型计算图内部完成，因此无需对输入数据进行额外的预处理。

计算图的输出与 [EfficientNMS_TRT](https://github.com/NVIDIA/TensorRT/tree/master/plugin/efficientNMSPlugin)插件及分割头的输出一致，最后一个节点的名称为 `detection_masks`，形状为 `[batch_size, max_proposals, mask_height, mask_width]`，数据类型为 float32。

### 构建 TensorRT 引擎

可以直接使用上一步生成的 ONNX 计算图，通过 `trtexec`构建 TensorRT 引擎。如果 `$PATH`中未包含 `trtexec`，通常可在 `/usr/bin/trtexec`找到该二进制文件（具体取决于你的 TensorRT 安装方式）。运行命令：

```
trtexec --onnx=./model/converted.onnx --saveEngine=./model/engine.trt --stronglyTyped
```

为方便起见，本仓库也提供了 `build_engine.py`脚本，该脚本专门针对 Detectron 2 Mask R-CNN R50-FPN 3x 引擎的构建进行了优化。运行 `python3 build_engine.py --help`可查看可用设置的详细信息。

#### 引擎基准测试

可选地，你可以使用 `trtexec`工具获取已构建引擎的执行时间信息，命令如下：

```
trtexec \
    --loadEngine=./model/engine.trt \
    --iterations=100 --avgRuns=100
```

推理基准测试将运行，并在控制台打印出 GPU 计算延迟时间。根据你的环境，你可能会看到类似如下的输出：

```
GPU Compute Time: min = 30.1864 ms, max = 37.0945 ms, mean = 34.481 ms, median = 34.4187 ms, percentile(99%) = 37.0945 ms
```

以下是使用 fp32 数据精度的示例结果。这些结果是在 RTX A5000 显卡和 TensorRT 8.6.1 环境下获得的。mAP 值是按照 [评估 mAP 指标](#evaluate-map-metric)中的说明，在 COCO val2017 数据集上评估得出的。

| **精度** | **延迟** | **bbox COCO mAP** | **segm COCO mAP** |
| -------- | -------- | ----------------- | ----------------- |
| fp32     | 25.89 ms | 0.402             | 0.368             |

## 推理

为了获得最佳性能，推理应在 C++ 应用程序中进行，并利用 CUDA Graphs 启动推理请求。此外，通过此流程构建的 TensorRT 引擎也可以在 [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)或 [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)中执行。

不过，为了方便快速测试已构建的 TensorRT 引擎，这里也提供了一个 Python 推理脚本。

### Python 推理

要在图像集上使用 TensorRT 执行目标检测，请运行：

```
python infer.py \
    --engine ./model/engine.trt \
    --input ./data/input \
    --det2_config ./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --output ./data/output
```

其中输入路径可以是单个图像文件，也可以是包含 jpg/png/bmp 图像的目录。

该脚本包含若干可选参数，包括：

- `--nms_threshold`：允许覆盖默认的第二次 NMS 分数阈值参数。

- `--iou_threshold`：允许设置掩码分割的 IoU 阈值，默认为 0.5。

检测结果将被写入指定的输出目录，包括一张可视化图像和一个针对每张处理过的输入图像生成的制表符分隔的结果文件。

#### 示例图像

https://drive.google.com/uc?export=view&id=1AOW9IXqjrU7eVYmaue-pqijNucXmx_s0

https://drive.google.com/uc?export=view&id=1m1fp2v41DOqKfj423G0-eyKVurrPNYGx

### 评估 mAP 指标

给定验证数据集（如 [COCO val2017 数据](http://images.cocodataset.org/zips/val2017.zip)），你可以获取已构建 TensorRT 引擎的 mAP 指标。这将使用 [Detectron 2 评估](https://github.com/facebookresearch/detectron2/tree/main/detectron2/evaluation)仓库中的 mAP 指标工具函数。请务必遵循 [使用内置数据集指南](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html)正确设置 COCO 或自定义数据集。此外，请在 `/datasets`目录所在的同一文件夹中运行 `eval_coco.py`，否则会出现以下错误：

```
FileNotFoundError: [Errno 2] No such file or directory: 'datasets/coco/annotations/instances_val2017.json'
```

要运行评估，请执行：

```
python eval_coco.py \
    --engine ./model/engine.trt \
    --input /root/cys/PROJECT/00-COMMON/DATA/DT2/coco/val2017 \
    --det2_config ./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --det2_weights /root/.torch/iopath_cache/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```

该脚本包含若干可选参数，包括：

- `--nms_threshold`：允许覆盖默认的第二次 NMS 分数阈值参数。

- `--iou_threshold`：允许设置掩码分割的 IoU 阈值，默认为 0.5。

mAP 指标对所使用的 NMS 分数阈值非常敏感，因为使用较高的阈值会降低模型的召回率，从而导致 mAP 值下降。建议为不同目的构建独立的 TensorRT 引擎。即，一个使用默认阈值（如 0.5）的引擎专门用于 mAP 验证，另一个使用应用特定阈值（如 0.8）的引擎用于实际部署。这也是我们在 `create_onnx.py`脚本中将 NMS 阈值保留为可配置参数的原因。
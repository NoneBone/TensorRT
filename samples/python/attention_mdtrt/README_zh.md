# 基于 TensorRT 的多设备注意力推理

 en [English](./README.md) ｜ zh_CN [简体中文](./README_zh.md)

本示例演示如何使用 TensorRT 的多设备推理功能在多个 GPU 上运行自注意力模型。内容包括作为基准的单 GPU 执行，以及使用 MPI 进行进程管理、NCCL 进行 GPU 间通信的多 GPU 执行。

## 简介

TensorRT 支持将单个模型拆分到多个 GPU 上进行推理。这在模型过大无法放入单张 GPU，或希望通过跨设备并行计算降低延迟时非常有用。

TensorRT 为多设备推理提供了多种并行策略，包括张量并行（TP）和上下文并行（CP）。本示例聚焦于 **上下文并行（CP）**，即输入序列沿序列维度拆分到不同 GPU 上。大多数操作（线性投影、归一化）由各 GPU 独立处理其对应片段，但注意力机制需要跨设备通信，因为每个 token 都需要关注所有其他 token。TensorRT 通过嵌入在模型中的集合通信操作透明地处理这一过程。关于上下文并行的详细解释，请参见 [Context Parallelism for Scalable Million-Token Inference](https://arxiv.org/abs/2411.01783)。

### 工作原理

要在多 GPU 上运行模型，必须对模型进行 **切分（shard）**，即拆分成各 GPU 可独立执行的片段。对于上下文并行，切分意味着将输入序列分配到各 GPU，并插入通信算子，使每个 GPU 仍能基于完整序列计算正确的注意力结果。

当模型为多设备执行进行切分时，特殊的 **DistCollective** 操作会被插入到 ONNX 图中：

- **ReduceScatter**：跨 GPU 拆分并归约输入数据。在起始阶段用于分发输入序列。

- **AllGather**：从所有 GPU 收集数据形成完整张量。在注意力计算前使用，使各 GPU 能看到完整的 K/V 张量；在输出投影后使用，以重建完整输出。

这些算子并不存在于单设备 ONNX 模型中。它们由 `polygraphy multi-device shard`工具根据描述模型切分方式的切分提示自动插入。

### 什么是 hint.json？

`hint.json`文件告诉 polygraphy 切分工具如何划分模型。切分工具读取单设备 ONNX 图，根据提示中指定的注意力层和 I/O 张量，在适当位置插入 DistCollective 算子。

```
{
    "parallelism": "CP",
    "attention_layers": [
        {
            "q": "q_scaled",          // 输入到 QK^T 矩阵乘法的 Q 张量名称
            "gather_kv": true,        // 在注意力计算前跨设备收集 K/V
            "gather_q": false         // Q 保持本地（各 GPU 持有自己的分片）
        }
    ],
    "dist_collectives": {
        "nb_rank": 2,                 // 切分所用的 GPU 数量
        "reduce_op": "max"            // ReduceScatter 使用的归约操作
    },
    "inputs": [
        {
            "name": "input",          // ONNX 模型中的输入张量名称
            "seq_len_idx": 0,         // 序列长度所在的维度索引
            "rank": 3                 // 维度数
        }
    ],
    "outputs": [
        {
            "name": "output",
            "seq_len_idx": 0,
            "rank": 3
        }
    ]
}
```

关于切分工具及 hint 格式的完整文档，请参阅 [Polygraphy 多设备文档](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/polygraphy/tools/multi_device)。

## 前置条件

- 配备多 GPU 的机器

- `polygraphy`>= 0.49.25（需支持 `multi-device shard`）

## 安装依赖

```
pip3 install -r requirements.txt
```

## 生成 ONNX 模型

### 步骤 1：生成单设备模型

```
python3 create_onnx.py --output attention_sd.onnx
```

这将创建一个自注意力模型，具有以下特性：

- 输入/输出：`(sequence_length, batch_size, 4096)`，float16 精度

- 32 个注意力头，每头 128 维

- 包含 Q/K/V 投影、RMSNorm、缩放点积注意力和输出投影

### 步骤 2：为多设备切分

```
polygraphy multi-device shard attention_sd.onnx -s hint.json -o attention_md.onnx
```

此命令根据 `hint.json`向模型中插入 DistCollective 算子（ReduceScatter、AllGather）。生成的 `attention_md.onnx`设计用于在 2 个 GPU 上运行。

## 运行示例

### 单 GPU

```
python3 attention_mdtrt.py \
  --onnx-path attention_sd.onnx \
  --sequence-length 56320 \
  --batch-size 1 \
  --num-iterations 50
```

### 多 GPU（2 个 GPU）

```
mpirun -np 2 python3 attention_mdtrt.py \
  --onnx-path attention_md.onnx \
  --sequence-length 56320 \
  --batch-size 1 \
  --num-iterations 50
```

### 使用指定的 libnccl.so

```
LD_PRELOAD=/path/to/libnccl.so mpirun -np 2 python3 attention_mdtrt.py \
  --onnx-path attention_md.onnx
```

## 许可证

使用、复制和分发的相关条款与条件，请参见 [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)文档。

## 变更日志

2026 年 4 月

新增 `create_onnx.py`，使用 GraphSurgeon Layer API 生成 ONNX 模型。新增 `--save-output`标志用于保存推理输出。更新文档，补充 DistCollective 和切分相关说明。

2026 年 1 月

本示例首次发布。

## 已知问题

无
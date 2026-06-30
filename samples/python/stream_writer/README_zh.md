# TensorRT Python 示例：流写入器

本示例演示如何使用 TensorRT Python API 通过 `IStreamWriter`接口将引擎直接序列化到自定义流中，而非文件或内存缓冲区。这对于需要控制引擎字节写入方式和位置的高级场景非常有用（例如，写入网络套接字、自定义缓冲区或内存流）。

## 本示例功能

- 构建一个包含两个卷积层和 ReLU 激活函数的简单 TensorRT 网络。

- 实现一个继承自 `trt.IStreamWriter`的自定义 `StreamWriter`类，用于收集序列化的引擎字节。

- 使用 `builder.build_serialized_network_to_stream()`序列化引擎，并将字节写入自定义流。

- 从收集的字节中反序列化引擎以验证正确性。

## 文件结构

- `build.py`：包含示例代码的主脚本。

- `README.md`：本文档。

## 运行方式

1. **安装依赖**

   确保已安装以下 Python 包：

   -    `tensorrt`

   -    `numpy`

   -    `polygraphy`

   可通过 pip 安装 Polygraphy：

   ```
   pip install polygraphy
   ```

   `tensorrt`Python 包通常由 NVIDIA 以 wheel 文件形式提供。

2. **运行示例**

   ```
   python3 build.py
   ```

   您将看到指示网络构建完成、引擎成功构建并序列化到流中，随后成功反序列化的输出信息。

## 核心概念

- **IStreamWriter**：TensorRT 中的一个接口，允许您定义写入序列化引擎字节的自定义逻辑。您必须实现 `write(self, data)`方法。

- **build_serialized_network_to_stream**：一个将网络序列化并将字节写入提供的 `IStreamWriter`实例的方法。

## 示例输出

```
Constructing network...
[I] TF32 默认禁用。开启 TF32 可在精度略有差异的情况下获得更佳性能。
[I] 配置配置文件：[
        配置文件 0：
            {输入 [最小=[1, 3, 224, 224], 最优=[1, 3, 224, 224], 最大=[1, 3, 224, 224]]}
    ]
构建引擎并序列化到流中...
写入流的总字节数为 267836
从流中反序列化引擎...
引擎反序列化成功
```

# 许可证

有关使用、复制和分发的条款和条件，请参阅 [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)文档。

# 更新日志

2025年9月

本示例首次发布。

# 已知问题

本示例中暂无已知问题。
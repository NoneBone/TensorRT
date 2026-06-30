# TensorRT 重构示例

本目录包含经过重构和改进的 TensorRT 示例，展示了最佳实践与现代实现方式。

## 可用示例

| 示例名称                                                     | 描述                                         | 格式    |
| ------------------------------------------------------------ | -------------------------------------------- | ------- |
| [1_run_onnx_with_tensorrt](./1_run_onnx_with_tensorrt) | 演示 ONNX 模型转换为 TensorRT 并进行推理对比 | `ipynb` |
| [2_construct_network_with_layer_apis](./2_construct_network_with_layer_apis) | 使用 TensorRT Layer API 构建网络             | `ipynb` |

## 启动说明

1. 进入目标示例目录并启动 Jupyter 服务：

   ```
   pip install notebook
   cd 1_run_onnx_with_tensorrt # 或任何其他示例
   jupyter notebook
   ```

2. 随后，在浏览器中打开的 Jupyter Notebook 界面里，点击打开 `main.ipynb`文件。

# 变更日志

2025 年 10 月

迁移至强类型（strongly typed）API。

2025 年 8 月

移除对 < 3.10 Python 版本的支持。

# Preparing sample onnx and data

See [Preparing sample data](../README.md#preparing-sample-data) in the main samples README.

# onnxslim

- 删除无用节点（Dead ends）
- 合并连续的 Conv / BN / ReLU
- 消除 Identity / Cast / Reshape 冗余
- 推断并固化静态 shape
- 降低算子复杂度

# polygraphy

- 修复非法 / 不规范的 ONNX 结构
- 统一数据类型 / 维度语义
- 移除可能导致 TRT 报错的“边缘情况”
- 把可计算的常量表达式直接算成常量（--fold-constants）

# cmd abstract
```sh
python -m onnxslim mnist.onnx mnist.sim.onnx
polygraphy surgeon sanitize mnist.sim.onnx \
  -o mnist.clean.onnx \
  --fold-constants

trtexec --onnx=mnist.onnx
origin: Throughput: 54549.2 qps Total Host Walltime: 3.00002 s Total GPU Compute Time: 2.56574 s

trtexec --onnx=mnist.clean.onnx
clean:  Throughput: 55862.5 qps Total Host Walltime: 2.99956 s Total GPU Compute Time: 2.52432 s
```

# full log
```sh
(cp312) [root@cys-cys ~/cys/PROJECT/00-COMMON/DEMO/02-TensorRT/samples/trtexec]# python -m onnxslim /root/cys/PROJECT/00-COMMON/DEMO/02-TensorRT/data/mnist/mnist.onnx /root/cys/PROJECT/00-COMMON/DEMO/02-TensorRT/data/mnist/mnist.sim.onnx
Warning: Onnx Runtime version 1.24 has no specified compatible ONNX version. Compatibility issues may occur.
+-----------------------+---------------------------+------------------------------------------+
|      Model Name       |        mnist.onnx         | /root/cys/PROJECT/00-COMMON/DEMO/02-Tens |
|                       |                           |      orRT/data/mnist/mnist.sim.onnx      |
+-----------------------+---------------------------+------------------------------------------+
|      Model Info       | Op Set: 8 / IR Version: 3 |        Op Set: 8 / IR Version: 3         |
+-----------------------+---------------------------+------------------------------------------+
|      IN: Input3       |  float32: (1, 1, 28, 28)  |         float32: (1, 1, 28, 28)          |
| OUT: Plus214_Output_0 |     float32: (1, 10)      |             float32: (1, 10)             |
+-----------------------+---------------------------+------------------------------------------+
|          Add          |             3             |                    0                     |
|         Conv          |             2             |                    2                     |
|         Gemm          |             0             |                    1                     |
|        MatMul         |             1             |                    0                     |
|        MaxPool        |             2             |                    2                     |
|         Relu          |             2             |                    2                     |
|        Reshape        |             2             |                    1                     |
+-----------------------+---------------------------+------------------------------------------+
|      Model Size       |         23.45 KB          |                 23.43 KB                 |
+-----------------------+---------------------------+------------------------------------------+
|     Elapsed Time      |                                4.52 s                                |
+-----------------------+---------------------------+------------------------------------------+
(cp312) [root@cys-cys ~/cys/PROJECT/00-COMMON/DEMO/02-TensorRT/samples/trtexec]# cd /root/cys/PROJECT/00-COMMON/DEMO/02-TensorRT/data/mnist
(cp312) [root@cys-cys ~/cys/PROJECT/00-COMMON/DEMO/02-TensorRT/data/mnist]# polygraphy surgeon sanitize mnist.sim.onnx \
  -o mnist.clean.onnx \
  --fold-constants
[W] 'colored' module is not installed, will not use colors when logging. To enable colors, please install the 'colored' module: python3 -m pip install colored
[I] RUNNING | Command: /opt/conda/envs/cp312/bin/polygraphy surgeon sanitize mnist.sim.onnx -o mnist.clean.onnx --fold-constants
[I] Loading model: /root/cys/PROJECT/00-COMMON/DATA/TRT_SAMPLES/mnist/mnist.sim.onnx
[I] Original Model:
    Name: CNTKGraph | ONNX Opset: 8
    
    ---- 1 Graph Input(s) ----
    {Input3 [dtype=float32, shape=(1, 1, 28, 28)]}
    
    ---- 1 Graph Output(s) ----
    {Plus214_Output_0 [dtype=float32, shape=(1, 10)]}
    
    ---- 7 Initializer(s) ----
    
    ---- 8 Node(s) ----
    
[I] Folding Constants | Pass 1
[I]     Total Nodes | Original:     8, After Folding:     8 |     0 Nodes Folded
[I] Saving ONNX model to: mnist.clean.onnx
[I] New Model:
    Name: CNTKGraph | ONNX Opset: 8
    
    ---- 1 Graph Input(s) ----
    {Input3 [dtype=float32, shape=(1, 1, 28, 28)]}
    
    ---- 1 Graph Output(s) ----
    {Plus214_Output_0 [dtype=float32, shape=(1, 10)]}
    
    ---- 7 Initializer(s) ----
    
    ---- 8 Node(s) ----
    
[I] PASSED | Runtime: 17.684s | Command: /opt/conda/envs/cp312/bin/polygraphy surgeon sanitize mnist.sim.onnx -o mnist.clean.onnx --fold-constants
```
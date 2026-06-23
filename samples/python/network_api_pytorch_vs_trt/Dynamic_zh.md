


# Brief

本样例基于 [NV-TRT样例](https://github.com/NVIDIA/TensorRT/tree/main/samples/python/network_api_pytorch_mnist)扩展如下功能：
1. 训练结果保存与快速加载：支持保存为 pth、onnx；支持读取以 pth、onnx、trt。
2. 推理阶段性能对比：pytorch vs trt，显存占用与耗时分解。
3. TRT[动态](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dynamic-shapes.html)批尺寸推理：分辨率固定为 28*28，批尺寸介于 1 到 1024。
4. TRT动态分辨率推理：批尺寸固定为 BATCH_SIZE，分辨率介于 1x1 到 1024x1024。

# Step

支持三种执行流程, 其中 ① 用于快速实现高性能；②等价于细粒度的控制 trtexec 转换；③作为对比的 baseline。

```sh
learn → PTH
        ├─→ ONNX ──→ TRT ───→ ① TRT-Infer
        ├───────────────────→ ② Online TRT-Infer
        └───────────────────→ ③ Py-Infer
```

# Run

NOTICE: `pip install -r requirements.txt`.

## 1. 动态批尺寸测试
```sh
# 读取 pth 以创建 onnx （ 缺失时，会训练 2 轮以输出pth ）
python ./sample.py --outOnnx 1
# onnx 转换为 trt
trtexec --onnx=mnist_fp32_dBS.onnx --saveEngine=mnist_fp32_dBS.trt --minShapes=input:1x1x28x28 --optShapes=input:128x1x28x28 --maxShapes=input:1024x1x28x28 --shapes=input:128x1x28x28

# py后端
python ./sample.py --use_exist 1 --use_trt 0 --bs 100
# trt后端
python ./sample.py --use_exist 1 --use_trt 1 --bs 100
# 动态 BS 测试脚本（两种后端对比）
./opt.sh 0
```
## 2. 动态分辨率测试

```sh
# 执行测试
python ./sample.py --use_exist 0 --use_trt 1 --shape 64
# 动态 shape 测试脚本
./opt.sh 1
```

# Question

## Q1 dynamic feature

动态批尺寸与分辨率分为 2 次实验，暂不支持同时进行。这是因为目前方案的缺点与优点：
- 缺点：在线引擎构建中，resize 操作的结果尺寸的 bs 维度，需要设置为实际工作的 bs 尺寸才能正确工作，这导致了动态特性丢失。详见源码`TODO`。
- 优点：只构建一次引擎，完成 resize+predict，将具备最低的启动开销，而不是官方 [DynamicShape](https://github.com/NVIDIA/TensorRT/tree/main/samples/sampleDynamicReshape) 样例的分为 2 个引擎来实现。
- 修复方向：拆分为 2 个独立引擎，以正确支持 2 种动态特性。

## Q2 quant support

待测试fp16, bf16, 参考[NV-Quant](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html).

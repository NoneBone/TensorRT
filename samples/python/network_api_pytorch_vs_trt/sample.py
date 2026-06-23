#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
# add
import argparse
import cv2

import os
# GPU_ID = 5
# os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
# set CUDA_VISIBLE_DEVICES before importing torch
from helper import *

# This sample uses an MNIST PyTorch model to create a TensorRT Inference Engine
import model
import numpy as np

import tensorrt as trt

sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
import common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class ModelData(object):
    INPUT_NAME = "input"
    INPUT_SHAPE = (-1, 1, -1, -1)
    OUTPUT_NAME = "output"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32

    MIN_RES = 1
    OPT_RES = 28
    MAX_RES = 1024

def populate_network(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(
        name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE
    )

    # REF: https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/operators/Resize.html#examples
    resize = network.add_resize(input_tensor)
    resize.resize_mode = trt.InterpolationMode.NEAREST
    # TODO: bs 设置为-1 会报错， bs 设置为 MAX_BS 会导致结果不太对。bs 目前必须得设置为 运行时实际bs, 导致了动态批尺寸并不支持
    resize.shape = (BATCH_SIZE, 1, ModelData.OPT_RES, ModelData.OPT_RES)
    resize.coordinate_transformation = trt.ResizeCoordinateTransformation.ALIGN_CORNERS
    resize.name = "input_resize"
    
    def add_matmul_as_fc(net, input, outputs, w, b):
        assert len(input.shape) >= 3
        m = 1 if len(input.shape) == 3 else input.shape[0]
        k = int(np.prod(input.shape) / m)
        assert np.prod(input.shape) == m * k
        n = int(w.size / k)
        assert w.size == n * k
        assert b.size == n

        input_reshape = net.add_shuffle(input)
        input_reshape.reshape_dims = trt.Dims2(m, k)

        filter_const = net.add_constant(trt.Dims2(n, k), w)
        mm = net.add_matrix_multiply(
            input_reshape.get_output(0),
            trt.MatrixOperation.NONE,
            filter_const.get_output(0),
            trt.MatrixOperation.TRANSPOSE,
        )

        bias_const = net.add_constant(trt.Dims2(1, n), b)
        bias_add = net.add_elementwise(
            mm.get_output(0), bias_const.get_output(0), trt.ElementWiseOperation.SUM
        )

        output_reshape = net.add_shuffle(bias_add.get_output(0))
        output_reshape.reshape_dims = trt.Dims4(m, n, 1, 1)
        return output_reshape
    
    conv1_w = weights["conv1.weight"].cpu().numpy()
    conv1_b = weights["conv1.bias"].cpu().numpy()
    conv1 = network.add_convolution_nd(
        input=resize.get_output(0), # input_tensor
        num_output_maps=20,
        kernel_shape=(5, 5),
        kernel=conv1_w,
        bias=conv1_b,
    )
    conv1.stride_nd = (1, 1)

    pool1 = network.add_pooling_nd(
        input=conv1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2)
    )
    pool1.stride_nd = trt.Dims2(2, 2)

    conv2_w = weights["conv2.weight"].cpu().numpy()
    conv2_b = weights["conv2.bias"].cpu().numpy()
    conv2 = network.add_convolution_nd(
        pool1.get_output(0), 50, (5, 5), conv2_w, conv2_b
    )
    conv2.stride_nd = (1, 1)

    pool2 = network.add_pooling_nd(conv2.get_output(0), trt.PoolingType.MAX, (2, 2))
    pool2.stride_nd = trt.Dims2(2, 2)

    fc1_w = weights["fc1.weight"].cpu().numpy()
    fc1_b = weights["fc1.bias"].cpu().numpy()
    fc1 = add_matmul_as_fc(network, pool2.get_output(0), 500, fc1_w, fc1_b)

    relu1 = network.add_activation(
        input=fc1.get_output(0), type=trt.ActivationType.RELU
    )

    fc2_w = weights["fc2.weight"].cpu().numpy()
    fc2_b = weights["fc2.bias"].cpu().numpy()
    fc2 = add_matmul_as_fc(
        network, relu1.get_output(0), ModelData.OUTPUT_SIZE, fc2_w, fc2_b
    )

    fc2.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=fc2.get_output(0))


def build_engine(weights, max_batch_size=1024, READ_EXIST=False):
    '''
    READ_EXIST 的启用，依赖于 trtexec 工具输出的 engine/trt。
    如果需要细粒度控制转换过程, 可以 
    ①采用 poly 工具编辑 onnx 后再转换; 
    ②基于权重构建 engine, 如 else 分支
    '''
    if READ_EXIST:
        # 仅用于动态批尺寸测试, 暂不支持 resize 操作
        nvTT.time_push("READ_EXIST")
        f = open(ONNX_PATH.replace("onnx", "trt"), "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())
        nvTT.time_pop()
        return engine
    else:
        # For more information on TRT basics, refer to the introductory samples.
        nvTT.time_push("create_network")

        builder = trt.Builder(TRT_LOGGER)
        network_creation_flag = 0
        if "EXPLICIT_BATCH" in trt.NetworkDefinitionCreationFlag.__members__.keys():
            network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        else:
            # pass
            network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        network = builder.create_network(network_creation_flag)
        config = builder.create_builder_config()
        runtime = trt.Runtime(TRT_LOGGER)

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))

        # Optimization Profile
        profile = builder.create_optimization_profile()
        #   min shape  (batch=1)
        profile.set_shape(
            ModelData.INPUT_NAME,
            min=(1, *ModelData.INPUT_SHAPE[1:2], ModelData.MIN_RES, ModelData.MIN_RES),          # (1,1,28,28)
            opt=(max(1, max_batch_size // 4), *ModelData.INPUT_SHAPE[1:2], ModelData.OPT_RES, ModelData.OPT_RES),
            max=(max_batch_size, *ModelData.INPUT_SHAPE[1:2], ModelData.MAX_RES, ModelData.MAX_RES),
        )
        profile.set_shape(
            "input_resize",
            min=(1, *ModelData.INPUT_SHAPE[1:2], ModelData.OPT_RES, ModelData.OPT_RES),
            opt=(max(1, max_batch_size // 4), *ModelData.INPUT_SHAPE[1:2], ModelData.OPT_RES, ModelData.OPT_RES),
            max=(max_batch_size, *ModelData.INPUT_SHAPE[1:2], ModelData.OPT_RES, ModelData.OPT_RES),
        )
        profile.set_shape(
            ModelData.OUTPUT_NAME,
            min=(1, ModelData.OUTPUT_SIZE),
            opt=(max(1, max_batch_size // 4), ModelData.OUTPUT_SIZE),
            max=(max_batch_size, ModelData.OUTPUT_SIZE),
        )
        config.add_optimization_profile(profile)

        # Populate the network using weights from the PyTorch model.
        nvTT.time_pop("create_network")
        nvTT.time_push("populate")
        populate_network(network, weights)
        nvTT.time_pop("populate")

        # Build and return an engine.
        nvTT.time_push("serial")
        plan = builder.build_serialized_network(network, config)
        nvTT.time_pop()
        nvTT.time_push("deSerial")
        deplan = runtime.deserialize_cuda_engine(plan)
        nvTT.time_pop()

        return deplan


def load_batch_testcase(model, host_buffer, batch, targetSize = 28):
    data_batch, label_batch = next(iter(model.test_loader))

    data_batch = data_batch[:batch]          # (batch, 1, 28, 28)
    label_batch = label_batch[:batch]        # (batch,)

    # Resize
    if(targetSize!=28):
        resized = np.empty((batch, 1, targetSize, targetSize), dtype=np.float32)
        for i in range(batch):
            img = data_batch[i].numpy().squeeze()          # (28, 28)
            img_resized = cv2.resize(img, (targetSize, targetSize), interpolation=cv2.INTER_LINEAR)
            resized[i, 0, :, :] = img_resized
        flat = resized.reshape(-1)   # float32, shape (batch*784,)
        print(f"[Info] input for trt is {resized.shape}")
    else:
        flat = data_batch.numpy().reshape(-1)   # float32, shape (batch*784,)
    np.copyto(host_buffer[: flat.size], flat)

    return label_batch.numpy()


# static para.
MAX_BS = 1024                       # 注意：测试集全部只有 1000 张图, 实际测试会 min 到 1000
PT_WEIGHTS = "mnist_fp32.pth"       # 训练参数详见 model.py 51行 + 89行
ONNX_PATH = "mnist_fp32_dynamic.onnx"

def main(args=None):
    parser = argparse.ArgumentParser(description="MNIST TensorRT Demo")
    parser.add_argument("--bs", type=int, default=1000, help="Inference batch size")
    parser.add_argument("--use_trt", type=int, default=1, help="1 for use trt banckend, 0 for pytorch.")
    parser.add_argument("--shape", type=int, default=28, help="Input resolution (both H and W).")
    parser.add_argument("--use_exist", type=int, default=0, help="1 for dynamic batch size test. 0 for dynamic resolution test.")
    parser.add_argument("--outOnnx", type=int, default=0, help="Exit after exporting the ONNX file")
    args = parser.parse_args(args)

    global BATCH_SIZE
    BATCH_SIZE = min(args.bs, 1000)
    SOURCE_RES = args.shape

    common.add_help(description="Runs an MNIST network using a PyTorch model")
    # Train the PyTorch model
    nvTT.time_push("t_allTime")
    # MP._record_memory_snapshot("init")
    mnist_model = model.MnistModel()

    if os.path.exists(PT_WEIGHTS):
        print(f"[Info] Found {PT_WEIGHTS}, loading weights...")
        mnist_model.load_weights(PT_WEIGHTS, device="cpu")
        nvTT.time_push("weight")
        weights = mnist_model.get_weights()
        nvTT.time_pop()
    else:
        print("[Info] No weights found, training from scratch...")
        nvTT.time_push("train")
        MP._record_memory_snapshot("bfTrain")
        mnist_model.learn()
        nvTT.time_pop("train")
        MP._record_memory_snapshot("bfWeight")
        nvTT.time_push("weight")
        mnist_model.save_weights(PT_WEIGHTS)
        nvTT.time_pop()
        print(f"[Info] Saved to {PT_WEIGHTS}.")

    if args.outOnnx:
        print("[Info] Exporting model to ONNX...")
        mnist_model.export_onnx(ONNX_PATH)
        import sys
        sys.exit()

    # Do inference.
    MP._record_memory_snapshot("afload")
    if not args.use_trt:
        print("[Info] inference backend: py ...")
        mnist_model.mytest(batch_size=BATCH_SIZE)
    else:
        print("[Info] inference backend: trt ...")
        engine = build_engine(weights, MAX_BS, READ_EXIST=args.use_exist)
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        # MP._record_memory_snapshot("afBuild")
        nvTT.time_push("allocate")
        inputs, outputs, bindings = common.allocate_buffers(engine, max_batch=MAX_BS, max_res=SOURCE_RES)
        # MP._record_memory_snapshot("afAlloc")

        context = engine.create_execution_context()
        nvTT.time_pop("allocate")

        concrete_shape = (BATCH_SIZE, 1, SOURCE_RES, SOURCE_RES)   # (BATCH_SIZE, 1, 28, 28)
        # Set the dynamic input shape for the execution context.
        context.set_input_shape(ModelData.INPUT_NAME, concrete_shape)

        # WARNING: 无效配置，不存在于 io_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        # concrete_reshape = (BATCH_SIZE, 1, ModelData.OPT_RES, ModelData.OPT_RES)   # (BATCH_SIZE, 1, 28, 28)
        # context.set_input_shape("input_resize", concrete_reshape) 
        
        nvTT.time_push("select")
        # case_num = load_random_test_case(mnist_model, pagelocked_buffer=inputs[0].host)
        labels = load_batch_testcase(
            mnist_model, host_buffer=inputs[0].host, batch=BATCH_SIZE, targetSize=SOURCE_RES,
        )
        nvTT.time_pop("select")
    
        # Use context manager for proper stream lifecycle management
        with common.CudaStreamContext() as stream:
            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            nvTT.time_push("inference")
            [output] = common.do_inference(# TODO: 按照 BS 拷贝 output，而不是完整的输出 MAX_BS
                context,
                engine=engine,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
            )
            nvTT.time_pop("inference")
            MP._record_memory_snapshot("afinf")

            nvTT.time_push("post")
            common.free_buffers(inputs, outputs)
            nvTT.time_pop("post")

        real_bs = min(min(MAX_BS, BATCH_SIZE), labels.size)
        output = output[:real_bs* ModelData.OUTPUT_SIZE]
        output.shape = (real_bs, ModelData.OUTPUT_SIZE)

        preds = np.argmax(output, axis=1)
        correct = int(np.sum(preds == labels))
        accuracy = correct / real_bs
        print(f"Batch Size: {real_bs}")
        print(f"Accuracy: {accuracy:.4f} ({correct}/{real_bs})")
        
    nvTT.time_pop("t_allTime")
    nvTT.save_print_time(pFlag=True)
    MP.print_summary()
    MP.reset()

    # nvTT.print_avg_stats()
    # PRINT_PEAK_MEM()


if __name__ == "__main__":
    nvTT.nvtx_start()
    main()
    nvTT.nvtx_stop()

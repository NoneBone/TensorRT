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

import os
import sys
import argparse

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
    INPUT_SHAPE = (-1, 1, 28, 28)
    OUTPUT_NAME = "output"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32


def populate_network(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(
        name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE
    )

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
        input=input_tensor,
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
    如果需要细粒度控制转换过程, 可以 ①TODO: 采用 poly 工具编辑 onnx 后再转换; ②基于权重构建 engine, 如 else 分支
    '''
    if READ_EXIST:
        nvTT.time_push("READ_EXIST")
        f = open("mnist_fp32_dBS.trt", "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())
        nvTT.time_pop()
        return engine
    else:
        # For more information on TRT basics, refer to the introductory samples.
        nvTT.time_push("create_network")

        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        config = builder.create_builder_config()
        runtime = trt.Runtime(TRT_LOGGER)

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))

        # Optimization Profile
        profile = builder.create_optimization_profile()
        #   min shape  (batch=1)
        profile.set_shape(
            ModelData.INPUT_NAME,
            min=(1, *ModelData.INPUT_SHAPE[1:]),          # (1,1,28,28)
            opt=(max(1, max_batch_size // 4), *ModelData.INPUT_SHAPE[1:]),  # 例如 8
            max=(max_batch_size, *ModelData.INPUT_SHAPE[1:]),               # (max,1,28,28)
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


def load_batch_testcase(model, host_buffer, batch):
    data_batch, label_batch = next(iter(model.test_loader))

    data_batch = data_batch[:batch]          # (batch, 1, 28, 28)
    label_batch = label_batch[:batch]        # (batch,)

    flat = data_batch.numpy().reshape(-1)   # float32, shape (batch*784,)
    np.copyto(host_buffer[: flat.size], flat)

    return label_batch.numpy()

# TRT
ONNX_OUTPUT = 0
# BATCH_SIZE = 1024 # 注意：测试集全部也只有 1000 张图
max_batch = 1024

MP.ONLY_TORCH_TENSOR = False
PT_WEIGHTS = "mnist_fp32.pth"
ONNX_PATH = "mnist_fp32_dBS.onnx"

def main(args=None):
    parser = argparse.ArgumentParser(description="MNIST TensorRT Demo")
    parser.add_argument("--bs", type=int, default=16, help="Inference batch size")
    parser.add_argument("--inf_back", type=int, default=1, help="1 for use trt banckend, 0 for pytorch.")
    args = parser.parse_args(args)
    global BATCH_SIZE, USE_TRT
    BATCH_SIZE = args.bs
    USE_TRT = args.inf_back

    common.add_help(description="Runs an MNIST network using a PyTorch model")
    # Train the PyTorch model
    nvTT.time_push("t_allTime")
    # MP._record_memory_snapshot("init")
    mnist_model = model.MnistModel()

    if os.path.exists(PT_WEIGHTS):
        print(f"Found {PT_WEIGHTS}, loading weights...")
        mnist_model.load_weights(PT_WEIGHTS, device="cpu")
        nvTT.time_push("weight")
        weights = mnist_model.get_weights()
        nvTT.time_pop()
    else:
        print("No weights found, training from scratch...")
        
        nvTT.time_push("train")
        MP._record_memory_snapshot("bfTrain")
        mnist_model.learn()
        nvTT.time_pop("train")
        MP._record_memory_snapshot("bfWeight")
        nvTT.time_push("weight")
        mnist_model.save_weights(PT_WEIGHTS)
        nvTT.time_pop()
        import sys
        sys.exit()

    # 开关控制 onnx 输出，用于另外的 快速 TRT 推理脚本
    if ONNX_OUTPUT:
        print("[Info] Exporting model to ONNX...")
        mnist_model.export_onnx(ONNX_PATH)
        import sys
        sys.exit()

    # Do inference.
    MP._record_memory_snapshot("afload")
    if not USE_TRT:
        print("[Info] inference backend: py ...")
        mnist_model.mytest(batch_size=BATCH_SIZE)
    else:
        print("[Info] inference backend: trt ...")
        engine = build_engine(weights, max_batch, READ_EXIST=False) # TODO: Fall back when not exist
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        # MP._record_memory_snapshot("afBuild")
        nvTT.time_push("allocate")
        inputs, outputs, bindings = common.allocate_buffers(engine, max_batch=max_batch)
        # MP._record_memory_snapshot("afAlloc")

        context = engine.create_execution_context()
        nvTT.time_pop("allocate")

        # context.set_binding_shape(0, (BATCH_SIZE, *ModelData.INPUT_SHAPE))
        concrete_shape = (BATCH_SIZE, *ModelData.INPUT_SHAPE[1:])   # (BATCH_SIZE, 1, 28, 28)

        # Set the dynamic input shape for the execution context.
        context.set_input_shape(ModelData.INPUT_NAME, concrete_shape)
        
        nvTT.time_push("select")
        # case_num = load_random_test_case(mnist_model, pagelocked_buffer=inputs[0].host)
        labels = load_batch_testcase(
            mnist_model, host_buffer=inputs[0].host, batch=BATCH_SIZE
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

        real_bs = min(min(max_batch, BATCH_SIZE), labels.size)
        output = output[:real_bs*10]
        output.shape = (real_bs, 10)

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

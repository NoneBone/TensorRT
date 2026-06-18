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
    INPUT_NAME = "data"
    INPUT_SHAPE = (-1, 1, 28, 28)
    OUTPUT_NAME = "prob"
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


def build_engine(weights, max_batch_size=1024):
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
USE_TRT = 1
BATCH_SIZE = 64
max_batch = 1024

USE_PYTORCH = not USE_TRT
MP.ONLY_TORCH_TENSOR = False
PT_WEIGHTS = "mnist_fp32.pth"
ONNX_PATH = "mnist_fp32.onnx"

def main():
    common.add_help(description="Runs an MNIST network using a PyTorch model")
    # Train the PyTorch model
    nvTT.time_push("t_allTime")
    MP._record_memory_snapshot("init")
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
        # TODO: 开关控制 onnx 输出，用于另外的 快速 TRT 推理脚本
        import sys
        sys.exit()

    # Do inference.
    MP._record_memory_snapshot("afload")
    if USE_PYTORCH:
        mnist_model.mytest(batch_size=1)
    else:
        engine = build_engine(weights, max_batch)
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        MP._record_memory_snapshot("afBuild")
        nvTT.time_push("allocate")
        inputs, outputs, bindings = common.allocate_buffers(engine, max_batch=max_batch)
        MP._record_memory_snapshot("afAlloc")

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

        if max_batch > BATCH_SIZE:
            output = output[:BATCH_SIZE*10]

        output.shape = (BATCH_SIZE, 10)
        preds = np.argmax(output, axis=1)
        correct = int(np.sum(preds == labels))
        accuracy = correct / BATCH_SIZE
        print(f"Batch Size: {BATCH_SIZE}")
        print(f"Accuracy: {accuracy:.4f} ({correct}/{BATCH_SIZE})")
        
    nvTT.time_pop("t_allTime")
    nvTT.save_print_time()


if __name__ == "__main__":
    nvTT.nvtx_start()
    main()
    nvTT.nvtx_stop()
    nvTT.print_avg_stats()
    PRINT_PEAK_MEM()
    MP.print_summary()

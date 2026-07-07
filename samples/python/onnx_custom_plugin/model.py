#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json

import wget
import onnx
import onnx_graphsurgeon as gs

MODEL_URL = "https://github.com/onnx/models/blob/main/validated/text/machine_comprehension/bidirectional_attention_flow/model/bidaf-9.onnx"

WORKING_DIR = os.environ.get("TRT_WORKING_DIR") or os.path.dirname(
    os.path.realpath(__file__)
)
MODEL_DIR = os.path.join(WORKING_DIR, "models")
RAW_MODEL_PATH = os.path.join(MODEL_DIR, "bidaf-9.onnx")
TRT_MODEL_PATH = os.path.join(MODEL_DIR, "bidaf-9-trt.onnx")


def _do_graph_surgery(raw_model_path, trt_model_path):
    graph = gs.import_onnx(onnx.load(raw_model_path))

    # Replace unsupported Hardmax with our CustomHardmax op
    hardmax_node = None
    for node in graph.nodes:
        if node.op == "Hardmax":
            node.op = "CustomHardmax"
            hardmax_node = node
    assert hardmax_node is not None, "Model does not contain a Hardmax node"

    # 原始的onnx模型还使用了另一个不支持的操作符 Compress 
    # “Compress”会返回第二个张量中所有评估为True的索引对应的第一个张量的值。
    # 在我们的例子中，第二个张量是 Hardmax 的输出，
    # 因此恰好只有一个索引会评估为 True，因为该位置的值为1，其余所有值均为0。
    # 我们可以通过对数值张量与 Hardmax 输出进行点积来实现与“Compress”相同的效果。
    # The original onnx model also uses another unsupported op called "Compress".
    # "Compress" returns values from the first tensor for all indices which evaluate to
    # True in the second tensor. In our case the second Tensor is the output of Hardmax,
    # so exactly one index will evaluate to true because the value at it will be 1, and
    # all other values will be 0. We can achieve the same result as "Compress" by taking the
    # dot product of our value tensor and the Hardmax output.
    #
    # 因此，我们将用子图 Einsum(Transpose_29, Hardmax) 替换掉子图 
    # Compress(Transpose_29, Cast(Reshape(Hardmax)))，其中 Einsum 中的方程采用点积运算。
    # So, we will replace the subgraph Compress(Transpose_29, Cast(Reshape(Hardmax)))
    # with the subgraph Einsum(Transpose_29, Hardmax) where the equation in Einsum takes the dot product.
    node_by_name = {node.name: node for node in graph.nodes}
    transpose_node = node_by_name["Transpose_29"]
    compress_node = node_by_name["Compress_31"]

    einsum_node = gs.Node(
        "Einsum",                               # ONNX 算子类型
        "Dot_of_Hardmax_and_Transpose",         # 节点名（调试用）
        attrs={"equation": "ij,ij->i"},         # 爱因斯坦求和记号
        inputs=[hardmax_node.outputs[0],        # 输入0：Hardmax 输出（掩码）
                transpose_node.outputs[0]],     # 输入1：Transpose_29 输出（数据）
        outputs=[compress_node.outputs[0]],     # 输出：直接复用 Compress 的输出张量
    )
    graph.nodes.append(einsum_node)
    
    # 将要被删除的旧子图与图清理操作（graph.cleanup()）分开处理
    # Separate the old subgraph which will be deleted with graph.cleanup()
    hardmax_node.o().inputs.clear()
    transpose_node.o().inputs.clear()
    compress_node.outputs.clear()

    # 同时移除将字符串转换为整数作为模型第一步的 CategoryMapper 节点。
    # Also remove the CategoryMapper nodes which convert strings to integers as the first step in the model.
    # We need to convert the following structure:
    #
    #      Input as                        Converted to
    #   String tokens                     Integer tokens
    #  ---------------->[CategoryMapper]------------------>[Rest of Model]
    #
    # into the following:
    #
    #      Input as
    #   Integer tokens
    #  ------------------>[Rest of Model]
    #
    # Later we will feed the model the integer tokens directly.
    # Note: list conversion is necessary because we modify graph.nodes in the for loop.
    category_mapper_nodes = [
        node for node in graph.nodes if node.op == "CategoryMapper"
    ]
    for node in category_mapper_nodes:
        # Remove CategoryMapper node from onnx graph
        graph.nodes.remove(node)

        # Also remove references its inputs in the graph's inputs
        for input_tensor in node.inputs:
            graph.inputs.remove(input_tensor)

        # The graph's new inputs are the Integer tokens output by CategoryMapper
        graph.inputs += node.outputs

        # Save String->Int map
        with open(node.name + ".json", "w") as fp:
            json.dump(node.attrs, fp)
    
    # 拓扑排序 + 清理
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), trt_model_path)


def make_trt_compatible_onnx_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(RAW_MODEL_PATH):
        wget.download(MODEL_URL, out=RAW_MODEL_PATH)
        print("\nDownloaded BiDAF model from Onnx Model Zoo")
    print("Performing graph surgery on Onnx Model Zoo BiDAF model")
    _do_graph_surgery(RAW_MODEL_PATH, TRT_MODEL_PATH)
    print("Graph Surgery complete!")


def main():
    if os.path.exists(TRT_MODEL_PATH):
        print("TRT-compatible onnx model already exists!")
    else:
        print("TRT-compatible onnx model not found, generating...")
        make_trt_compatible_onnx_model()


if __name__ == "__main__":
    main()

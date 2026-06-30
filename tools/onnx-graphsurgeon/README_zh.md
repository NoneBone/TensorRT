# ONNX GraphSurgeon

## 目录

- [简介](#introduction)

- [安装](#installation)

- [示例](#examples)

- [基础概念](#understanding-the-basics)

  -   [导入器](#importers)

  -   [中间表示 (IR)](#ir)

    -     [张量](#tensor)

    -     [节点](#node)

    -     [关于修改输入和输出的注意事项](#a-note-on-modifying-inputs-and-outputs)

    -     [计算图](#graph)

  -   [导出器](#exporters)

- [进阶用法](#advanced)

  -   [处理带外部数据的模型](#working-with-models-with-external-data)

## 简介

ONNX GraphSurgeon 是一个用于创建和修改 ONNX 模型的 Python 库。

如果您更倾向于使用图形界面，可以尝试 [Nsight DL Designer](https://developer.nvidia.com/nsight-dl-designer)，它集成了 ONNX GraphSurgeon 并提供了多种额外功能。

## 安装

### 使用预构建的 Wheel 包

```
python3 -m pip install onnx_graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
```

### 从源码构建

#### 使用 Make 命令

```
make install
```

#### 手动构建

1. 构建 Wheel 包：

```
make build
```

1. 在仓库**外部**手动安装 Wheel 包：

```
python3 -m pip install onnx-graphsurgeon/dist/onnx_graphsurgeon-*-py2.py3-none-any.whl
```

## 示例

[examples](./examples)目录包含了多个展示 ONNX GraphSurgeon 常见使用场景的示例。

提供的可视化结果均使用 [Netron](https://github.com/lutzroeder/netron)生成。

## 基础概念

ONNX GraphSurgeon 主要由三个部分组成：导入器、中间表示 (IR) 和导出器。

### 导入器

导入器用于将计算图导入到 ONNX GraphSurgeon 的中间表示中。导入器接口定义在 [base_importer.py](./onnx_graphsurgeon/importers/base_importer.py)中。

ONNX GraphSurgeon 还提供了[高层导入 API](./onnx_graphsurgeon/api/api.py)以便于使用：

```
graph = gs.import_onnx(onnx.load("model.onnx"))
```

### 中间表示 (IR)

中间表示 (IR) 是所有计算图修改发生的地方。它也可以用于从头创建新的计算图。IR 包含三个组成部分：[张量](./onnx_graphsurgeon/ir/tensor.py)（Tensor）、[节点](./onnx_graphsurgeon/ir/node.py)（Node）和 [计算图](./onnx_graphsurgeon/ir/graph.py)（Graph）。

几乎每个组件的所有成员变量都可以自由修改。关于这些类的各种属性的详细信息，您可以在交互式 Shell 中使用 `help(<class_or_instance>)`，或在脚本中使用 `print(help(<class_or_instance>))`来查看帮助输出，其中 `<class_or_instance>`是 ONNX GraphSurgeon 的类型或其实例。

#### 张量

张量分为两个子类：`Variable`（变量）和 `Constant`（常量）。

- `Constant`（常量）是指其值在事前已知，可以作为 NumPy 数组获取和修改的张量。

  *注意：`Constant`的 `values`属性是按需加载的。如果该属性未被访问，其值将不会作为 NumPy 数组加载。*

- `Variable`（变量）是指其值在推理时才可知，但可能包含数据类型和形状信息的张量。

张量的输入和输出始终是节点。

**来自 ResNet50 的一个常量张量示例：**

```
>>> print(tensor)
Constant (gpu_0/res_conv1_bn_s_0)
[0.85369843 1.1515082  0.9152944  0.9577646  1.0663182  0.55629414
 1.2009839  1.1912311  2.2619808  0.62263143 1.1149117  1.4921428
 0.89566356 1.0358194  1.431092   1.5360111  1.25086    0.8706703
 1.2564877  0.8524589  0.9436758  0.7507614  0.8945271  0.93587166
 1.8422242  3.0609846  1.3124607  1.2158023  1.3937513  0.7857263
 0.8928106  1.3042281  1.0153942  0.89356416 1.0052011  1.2964457
 1.1117343  1.0669073  0.91343874 0.92906713 1.0465593  1.1261675
 1.4551278  1.8252873  1.9678202  1.1031747  2.3236883  0.8831993
 1.1133649  1.1654979  1.2705412  2.5578163  0.9504889  1.0441847
 1.0620039  0.92997414 1.2119316  1.3101407  0.7091761  0.99814713
 1.3404484  0.96389204 1.3435135  0.9236031 ]
```

**来自 ResNet50 的一个变量张量示例：**

```
>>> print(tensor)
Variable (gpu_0/data_0): (shape=[1, 3, 224, 224], dtype=float32)
```

#### 节点

`Node`（节点）定义了计算图中的操作。节点可以指定属性；属性值可以是任何 Python 基本类型，以及 ONNX GraphSurgeon 的 `Graph`或 `Tensor`。

节点的输入和输出始终是张量。

**来自 ResNet50 的一个 ReLU 节点示例：**

```
>>> print(node)
 (Relu)
    Inputs: [Tensor (gpu_0/res_conv1_bn_1)]
    Outputs: [Tensor (gpu_0/res_conv1_bn_2)]
```

在此示例中，节点没有属性。否则，属性将以 `OrderedDict`的形式显示。

#### 关于修改输入和输出的注意事项

节点和张量的 `inputs`（输入）/`outputs`（输出）成员具有特殊逻辑：当您进行更改时，它会更新所有受影响节点/张量的输入/输出。这意味着，例如，当您修改某个张量的 `outputs`时，**不需要**更新其对应节点的 `inputs`。

考虑以下节点：

```
>>> print(node)
 (Relu).
    Inputs: [Tensor (gpu_0/res_conv1_bn_1)]
    Outputs: [Tensor (gpu_0/res_conv1_bn_2)]
```

可以这样访问输入张量：

```
>>> tensor = node.inputs[0]
>>> print(tensor)
Tensor (gpu_0/res_conv1_bn_1)
>>> print(tensor.outputs)
[ (Relu).
	Inputs: [Tensor (gpu_0/res_conv1_bn_1)]
	Outputs: [Tensor (gpu_0/res_conv1_bn_2)]
]
```

如果我们从张量的输出中移除该节点，这一变化也会反映在节点的输入中：

```
>>> del tensor.outputs[0]
>>> print(tensor.outputs)
[]
>>> print(node)
 (Relu).
    Inputs: []
    Outputs: [Tensor (gpu_0/res_conv1_bn_2)]
```

#### 计算图

`Graph`（计算图）包含零个或多个 `Node`（节点）以及输入/输出 `Tensor`（张量）。

中间张量不会被显式跟踪，而是从计算图包含的节点中检索得到。

`Graph`类公开了多个函数，此处列举一小部分：

- `cleanup()`：移除计算图中未使用的节点和张量。

- `toposort()`：对计算图进行拓扑排序。

- `tensors()`：返回一个 `Dict[str, Tensor]`，将张量名称映射到张量，通过遍历计算图中的所有张量实现。这是一个 `O(N)`操作，因此在大型计算图上可能会比较慢。

要查看完整的 Graph API，您可以在交互式 Python Shell 中执行 `help(onnx_graphsurgeon.Graph)`。

### 导出器

导出器用于将 ONNX GraphSurgeon 的中间表示导出为 ONNX 或其他类型的计算图。导出器接口定义在 [base_exporter.py](./onnx_graphsurgeon/exporters/base_exporter.py)中。

ONNX GraphSurgeon 还提供了[高层导出 API](./onnx_graphsurgeon/api/api.py)以便于使用：

```
onnx.save(gs.export_onnx(graph), "model.onnx")
```

## 进阶用法

### 处理带外部数据的模型

在 ONNX-GraphSurgeon 中使用带有外部存储数据的模型，与处理不带外部数据的 ONNX 模型几乎相同。有关如何加载此类模型的详细信息，请参阅 [官方 ONNX 文档](https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md#loading-an-onnx-model-with-external-data)。要将模型导入 ONNX-GraphSurgeon，您可以像往常一样使用 `import_onnx`函数。

在导出过程中，您只需额外执行一个步骤：

1. 像往常一样从 ONNX-GraphSurgeon 导出模型：

   ```
   model = gs.export_onnx(graph)
   ```

2. 更新模型，使其将数据写入外部位置。如果未指定位置，则默认与 ONNX 模型位于同一目录下：

   ```
   from onnx.external_data_helper import convert_model_to_external_data
   
   convert_model_to_external_data(model, location="model.data")
   ```

3. 然后您可以像往常一样保存模型：

   ```
   onnx.save(model, "model.onnx")
   ```
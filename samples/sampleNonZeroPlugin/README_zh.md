# NonZero Plugin for TensorRT using IPluginV3

**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
	* [Implementing a NonZero plugin using IPluginV3 interface](#implementing-a-nonzero-plugin-using-ipluginv3-interface)
	* [Creating network and building the engine](#creating-network-and-building-the-engine)
	* [Running inference](#running-inference)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## 说明

> 插件概述：（插件注册 → 创建 → 插入 → ）构建 → 运行， 这五个阶段分别由 TensorRT 的 PluginRegistry、IBuilder、INetworkDefinition、IExecutionContext 完成。

本示例 `sampleNonZeroPlugin` 实现了一个 **NonZero** 操作的插件，可配置为按**行序**（每行存放一组索引）或**列序**（每列存放一组索引）两种格式输出非零元素的索引。

NonZero 操作用于找出输入张量中所有非零元素的索引。



## 本示例的工作原理

本示例创建并运行一个 TensorRT 引擎，该引擎构建自一个仅包含单个 `NonZeroPlugin` 节点的网络。它演示了如何实现**输出形状依赖于输入数据值**的自定义层，并将其添加到 TensorRT 网络中。

具体而言，本示例涵盖以下步骤：
- #implementing-a-nonzero-plugin-using-ipluginv3-interface
- #creating-network-and-building-the-engine
- #running-inference

### 使用 IPluginV3 接口实现 NonZero 插件

在引入 `IPluginV3`（及其关联接口）之前，TensorRT 插件的输出形状只能依赖于**输入形状**，而不能依赖于**输入数据的值**。`IPluginV3OneBuild` 为 `IPluginV3` 暴露了构建期能力，提供了对此类**数据依赖型输出形状**的支持。

本示例中的 `NonZeroPlugin` 设计为处理形状为 $R \times C$ 的 **2-D 输入张量**。假设张量中包含 $K$ 个非零元素，且要求以**行序**方式输出非零索引（每组索引各占一行），则输出形状为 $K \times 2$。

输出形状通过 `IPluginV3OneBuild::getOutputShapes()` API 告知 TensorRT 构建器。其中第二维的表达很直观：

```cpp
outputs[0].d[1] = exprBuilder.constant(2);
```

插件中**每一个数据依赖维度的范围**都必须通过一个 ***size tensor***（尺寸张量）来表达。size tensor 是一个 `DataType::kINT32` 或 `DataType::kINT64` 类型的**标量输出**，必须作为插件的一个输出被添加。在本例中，只需声明一个 size tensor 来表示非零索引输出**第一维的范围**即可。声明 size tensor 时，需要提供其范围的上界（upper-bound）和最优估计值（optimum），两者均以 `IDimensionExpr` 的形式给出，可通过传入 `IPluginV3OneBuild::getOutputShapes()` 方法的 `IExprBuilder` 参数来构建：

- 对于未知输入，上界就是输入的总元素个数：
  ```cpp
  auto upperBound = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[0], *inputs[0].d[1]);
  ```
- 一个合理的 opt 估计值是"约一半元素为非零"：
  ```cpp
  auto optValue = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *upperBound, *exprBuilder.constant(2));
  ```

现在可以使用 `IExprBuilder::declareSizeTensor()` 方法声明 size tensor，该方法还需要指定该 size tensor 放置在哪个输出索引位置上。将其放在非零索引输出之后：

```cpp
auto numNonZeroSizeTensor = exprBuilder.declareSizeTensor(1, *optValue, *upperBound);
```

接下来就可以指定非零索引输出的第一维范围了：

```cpp
outputs[0].d[0] = numNonZeroSizeTensor;
```

别忘了将 size tensor 本身声明为**标量**（0-D）：

```cpp
outputs[1].nbDims = 0;
```

`NonZeroPlugin` 还可以通过插件属性 `rowOrder` 配置为以**列序**方式输出非零索引——将该属性设为 `0` 即可。此时插件的第一路输出形状变为 $2 \times K$，输出形状的指定也需要相应调整。

### 创建网络并构建引擎

要将插件加入网络，必须使用 `INetworkDefinition::addPluginV3()` 方法。

类似于 V2 插件使用的 `IPluginCreator`，V3 插件也必须注册一个实现了 `IPluginCreatorV3One` 接口的 **plugin creator**。

### 运行推理

示例输入从 MNIST 数据集中选取随机图像，并缩放至 `[0,1]` 区间。网络会同时输出**非零索引**以及**非零元素个数**。

## 前置条件

1. 准备示例数据

参见主示例 README 中的../README.md#preparing-sample-data。

## 运行示例

1. 按照 https://github.com/NVIDIA/TensorRT/ 中的编译说明编译本示例，或者使用如下指令快速构建本项目:

    ```sh
    rm -rf build
    mkdir build && cd build

    # build 目录下
    cmake .. \
      -DTRT_LIB_DIR=/root/cys/DRIVER/TensorRT-11.0.0.114/lib \
      -DTRT_INCLUDE_DIR=/root/cys/DRIVER/TensorRT-11.0.0.114/include \
      -DTRT_OUT_DIR=`pwd`/out \
      -DCMAKE_BUILD_TYPE=Debug \
      -DBUILD_SAMPLES=ON \
      -DBUILD_PARSERS=OFF \
      -DBUILD_PLUGINS=OFF

    rm ./out/sample_non_zero_plugin
    cmake --build . --target sample_non_zero_plugin -j$(nproc)
    ./out/sample_non_zero_plugin
    ```

2. 运行示例，从 ONNX 模型构建并运行 MNIST 引擎：

   ```
   ./sample_non_zero_plugin [-h or --help] [-d or --datadir=<path to data directory>] [--columnOrder] [--fp16]
   ./build/out/sample_non_zero_plugin -d ./data/mnist --columnOrder --fp16
   ```

3. 验证示例是否成功运行。运行成功后，输出应类似如下内容：
  对于 output 字段的 i,j，表示第 i 行第 j 列的非零元素的位置。
  本样例展示了，输出数据的 shape，与输入数据的内容有关。

   ```
   &&&& RUNNING TensorRT.sample_non_zero_plugin # ./sample_non_zero_plugin
   ...
   [I] Input:
   [I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
   [I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
   [I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.854902, 0
   [I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.858824, 0, 0, 0.0745098, 0, 0.564706, 0
   [I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.317647, 0, 0, 0.47451, 0, 0, 0
   [I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0431373, 0, 0, 0
   [I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
   [I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
   [I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.854902, 0, 0, 0.145098
   [I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.564706, 0, 0, 0.996078
   [I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.282353
   [I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
   [I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
   [I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.854902
   [I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.854902, 0, 0, 0.145098, 0, 0.564706
   [I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.564706, 0, 0, 0.996078, 0, 0
   [I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.282353, 0, 0
   [I]
   [I] Output:
   [I] 2 14
   [I] 3 9
   [I] 3 12
   [I] 3 14
   [I] 4 9
   [I] 4 12
   [I] 5 12
   [I] 8 12
   [I] 8 15
   [I] 9 12
   [I] 9 15
   [I] 10 15
   [I] 13 15
   [I] 14 10
   [I] 14 13
   [I] 14 15
   [I] 15 10
   [I] 15 13
   [I] 16 13
   &&&& PASSED TensorRT.sample_non_zero_plugin # ./sample_non_zero_plugin
   ```

### 示例 `--help` 选项

要查看所有可用命令行选项及说明，使用 `-h` 或 `--help` 参数即可。

# 其他参考资料

以下资源有助于更深入地理解 V3 TensorRT 插件及 NonZero 操作：

**NonZero**
- https://onnx.ai/onnx/operators/onnx__NonZero.html

**TensorRT 插件**
- https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending

**其他文档**
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html

# 许可证

关于使用、复制和分发的条款与条件，请参阅 https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html 文档。

# 变更记录

**2025 年 10 月**
迁移至强类型（strongly typed）API。

**2024 年 3 月**
本 `README.md` 文件的初始版本。

# 已知问题

Windows 用户若在 Visual Studio 中使用与 TensorRT 包所带版本不同的 CUDA 版本来编译本示例，需要通过 `Build Dependencies -> Build Customization` 菜单将项目**重定向**到已安装 CUDA 版本下进行构建。
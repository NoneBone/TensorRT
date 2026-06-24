/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! sampleNonZeroPlugin.cpp
//! This file contains a sample demonstrating a plugin for NonZero.
//! It can be run with the following command line:
//! Command: ./sample_non_zero_plugin [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//!

// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "nonZeroKernel.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string_view>
using namespace nvinfer1;

std::string const kSAMPLE_NAME = "TensorRT.sample_non_zero_plugin";

using half = __half;

void nonZeroIndicesHelper(nvinfer1::DataType type, void const* X, void* indices, void* count, void const* K, int32_t R,
    int32_t C, bool rowOrder, cudaStream_t stream)
{
    // 内部类型判断与转发，不支持类型会断言失败
    if (type == nvinfer1::DataType::kFLOAT)
    {
        nonZeroIndicesImpl<float>(static_cast<float const*>(X), static_cast<int32_t*>(indices),
            static_cast<int64_t*>(count), static_cast<int64_t const*>(K), R, C, rowOrder, stream);
    }
    else if (type == nvinfer1::DataType::kHALF)
    {
        nonZeroIndicesImpl<half>(static_cast<half const*>(X), static_cast<int32_t*>(indices),
            static_cast<int64_t*>(count), static_cast<int64_t const*>(K), R, C, rowOrder, stream);
    }
    else
    {
        ASSERT(false && "Unsupported data type");
    }
}

class NonZeroPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime
{// 多重继承（multiple inheritance）
public:
    // 默认拷贝构造函数：对类中 所有非 static成员​ 执行逐成员拷贝
    // 括号内为：常量左值引用（拷贝源）；default: 显式要求编译器生成默认实现；
    NonZeroPlugin(NonZeroPlugin const& p) = default;

    // 带参数的构造函数
    NonZeroPlugin(bool rowOrder)
        : mRowOrder(rowOrder)
    {
        initFieldsToSerialize();
    }

    void initFieldsToSerialize()
    {
        mDataToSerialize.clear();
        mDataToSerialize.emplace_back(PluginField("rowOrder", &mRowOrder, PluginFieldType::kINT32, 1));
        mFCToSerialize.nbFields = mDataToSerialize.size();
        mFCToSerialize.fields = mDataToSerialize.data();
    }

    // IPluginV3 methods
    // 根据调用方请求的插件能力类型（PluginCapabilityType），返回对应的能力接口指针；
    // 本质上是通过 this的多重继承关系完成接口分发。
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        try
        {
            if (type == PluginCapabilityType::kBUILD)
            {
                return static_cast<IPluginV3OneBuild*>(this);
            }
            if (type == PluginCapabilityType::kRUNTIME)
            {
                return static_cast<IPluginV3OneRuntime*>(this);
            }
            ASSERT(type == PluginCapabilityType::kCORE);
            return static_cast<IPluginV3OneCore*>(this);
        }
        catch (std::exception const& e)
        {
            sample::gLogError << e.what() << std::endl;
        }
        return nullptr;
    }

    // 创建当前插件对象的一个深拷贝（clone），并初始化字段，并将所有权交给 TensorRT
    // noexcept 不允许抛异常; override 重写基类虚函数
    // make_unique:	在堆上创建新对象, 比 new 的优点在于：即使后面 initFieldsToSerialize()抛异常，也不会泄漏。
    // auto 推导为智能指针
    // release: 取出 unique_ptr内部的裸指针, 释放所有权（不再自动 delete）,返回给 TensorRT
    IPluginV3* clone() noexcept override
    {
        auto clone = std::make_unique<NonZeroPlugin>(*this);
        clone->initFieldsToSerialize();
        return clone.release();
    }

    // IPluginV3OneCore methods
    char const* getPluginName() const noexcept override
    {
        return "NonZeroPlugin";
    }

    char const* getPluginVersion() const noexcept override
    {
        return "0";
    }

    char const* getPluginNamespace() const noexcept override
    {
        return "";
    }

    // IPluginV3OneBuild methods
    int32_t getNbOutputs() const noexcept override
    {
        return 2;
    }

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override
    {
        return 0;
    }

    // 告诉 TRT 在给定位置 pos上的输入或输出张量，是否支持当前的格式（format）+ 数据类型（data type）组合
    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        bool typeOk{false};
        if (pos == 0)
        {
            typeOk = inOut[0].desc.type == DataType::kFLOAT || inOut[0].desc.type == DataType::kHALF;
        }
        else if (pos == 1)
        {
            typeOk = inOut[1].desc.type == DataType::kINT32;
        }
        else // pos == 2
        {
            // size tensor outputs must be NCHW INT64
            typeOk = inOut[2].desc.type == DataType::kINT64;
        }

        return inOut[pos].desc.format == PluginFormat::kLINEAR && typeOk;
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        outputTypes[0] = DataType::kINT32;
        outputTypes[1] = DataType::kINT64;
        return 0;
    }

    // 根据输入张量的动态形状，推断并声明 NonZero 插件的输出张量形状，同时给出一个“合理的非零元素数量范围”，供 TensorRT 做内存规划
    // 
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override
    {
        // The input tensor must be 2-D
        if (inputs[0].nbDims != 2)
        {
            return -1;
        }

        outputs[0].nbDims = 2;
        // 计算输入元素总数的上界
        auto upperBound = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[0], *inputs[0].d[1]);

        // On average, we can assume that half of all elements will be non-zero
        auto optValue = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *upperBound, *exprBuilder.constant(2));
        // 尺寸张量声明，上下界与典型值
        auto numNonZeroSizeTensor = exprBuilder.declareSizeTensor(1, *optValue, *upperBound);
        if (!mRowOrder)
        {
            outputs[0].d[0] = exprBuilder.constant(2);
            outputs[0].d[1] = numNonZeroSizeTensor;
        }
        else
        {
            outputs[0].d[0] = numNonZeroSizeTensor;
            outputs[0].d[1] = exprBuilder.constant(2);
        }
        // 实际非零元素数量，是标量（size tensor）必须设置为 0 维
        // output at index 1 is a size tensor, size tensors must be declared as 0-D
        outputs[1].nbDims = 0;

        return 0;
    }

    // IPluginV3OneRuntime methods
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {

        int32_t const R = inputDesc[0].dims.d[0];
        int32_t const C = inputDesc[0].dims.d[1];

        auto type = inputDesc[0].type;

        if (!(type == nvinfer1::DataType::kHALF || type == nvinfer1::DataType::kFLOAT))
        {
            sample::gLogError << "Unsupported: Sample only supports DataType::kHALF and DataType::FLOAT" << std::endl;
            return -1;
        }

        cudaMemsetAsync(outputs[1], 0, sizeof(int64_t), stream);

        if (!mRowOrder && workspace == nullptr)
        {
            sample::gLogError << "Unsupported: workspace is needed but is null" << std::endl;
            return -1;
        }

        if (!mRowOrder)
        {
            // 在构建列主输出时，内核需要知道非零元素的总数，以便将非零索引写入正确的位置。
            // 因此，我们将两次启动内核：首先仅用于计算非零元素的总数，并将其存储在 workspace 中
            // 然后将非零索引实际写入输出缓冲区[0]。
            // When constructing a column major output, the kernel needs to be aware of the total number of non-zero
            // elements so as to write the non-zero indices at the correct places. Therefore, we will launch the kernel
            // twice: first, only to calculate the total non-zero count, which will be stored in workspace; and
            // then to actually write the non-zero indices to the outputs[0] buffer.
            cudaMemsetAsync(workspace, 0, sizeof(int64_t), stream);
            nonZeroIndicesHelper(type, inputs[0], nullptr, workspace, nullptr, R, C, mRowOrder, stream);
            nonZeroIndicesHelper(type, inputs[0], outputs[0], outputs[1], workspace, R, C, mRowOrder, stream);
        }
        else
        {
            nonZeroIndicesHelper(type, inputs[0], outputs[0], outputs[1], 0, R, C, mRowOrder, stream);
        }

        return 0;
    }
    // 当输入张量形状发生变化时，TensorRT 通知插件“形状变了”，插件可以在这里做响应
    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override
    {
        return 0;
    }
    // 为每个执行上下文（Execution Context）创建一个插件副本, 以避免数据竞争
    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override
    {
        return clone();
    }
    // 序列化这个插件时，需要保存哪些字段
    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        return &mFCToSerialize;
    }
    // 告诉 TRT：这个插件在执行时需要多少临时显存（workspace for kernel execution）
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override
    {
        return sizeof(int64_t);
    }

private:
    bool mRowOrder{true};
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

class NonZeroPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    NonZeroPluginCreator()
    {
        mPluginAttributes.clear();
        mPluginAttributes.emplace_back(PluginField("rowOrder", nullptr, PluginFieldType::kINT32, 1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    char const* getPluginName() const noexcept override
    {
        return "NonZeroPlugin";
    }

    char const* getPluginVersion() const noexcept override
    {
        return "0";
    }

    PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mFC;
    }

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override
    {
        try
        {
            using namespace std::string_view_literals;
            bool rowOrder{true};
            for (int32_t i = 0; i < fc->nbFields; ++i)
            {
                std::string_view const fieldName(fc->fields[i].name);
                if (fieldName == "rowOrder"sv)
                {
                    rowOrder = *static_cast<bool const*>(fc->fields[i].data);
                }
            }
            return new NonZeroPlugin(rowOrder);
        }
        catch (std::exception const& e)
        {
            sample::gLogError << e.what() << std::endl;
        }
        return nullptr;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return "";
    }

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
};

// 定义了一个 匿名命名空间（unnamed namespace），其中包含一个 结构体 NonZeroParams
// 该结构体 公有继承​ 自 samplesCommon::SampleParams，并新增了一个带默认值的成员 rowOrder。
namespace
{
struct NonZeroParams : public samplesCommon::SampleParams
{
    bool rowOrder{true};
};
} // namespace

// SampleNonZeroPlugin 解析入口参数，执行 build 和 infer，以及其他处理。

//! \brief  The SampleNonZeroPlugin class implements a NonZero plugin
//!
//! \details The plugin is able to output the non-zero indices in row major or column major order
//!
class SampleNonZeroPlugin
{
public:
    SampleNonZeroPlugin(NonZeroParams const& params)
        : mParams(params)
        , mRuntime(nullptr)
        , mEngine(nullptr)
    {
        mSeed = static_cast<uint32_t>(time(nullptr));
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:
    NonZeroParams mParams; //!< The parameters for the sample.

    //! The PluginCreator instance used to create NonZeroPlugin.
    //! The instance needs to stay alive across build and infer stages so that the entry in PluginRegistry remains valid
    //! throughout.
    NonZeroPluginCreator mPluginCreator;

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    uint32_t mSeed{};

    //!
    //! \brief Creates a TensorRT network and inserts a NonZero plugin
    //!
    bool constructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
        std::unique_ptr<nvinfer1::INetworkDefinition>& network, std::unique_ptr<nvinfer1::IBuilderConfig>& config);

    //!
    //! \brief Reads the input and stores the result in a managed buffer
    //!
    bool processInput(samplesCommon::BufferManager const& buffers);

    //!
    //! \brief Verifies the result
    //!
    bool verifyOutput(samplesCommon::BufferManager const& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates a network containing a NonZeroPlugin and builds
//!          the engine that will be used to run the plugin (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool SampleNonZeroPlugin::build()
{
    // getTRTLogger 获取一个 TensorRT 日志记录器对象，类型是 nvinfer1::ILogger&
    // createInferBuilder 返回裸指针 IBuilder*，并用 智能指针​ 接管 裸指针: 独占所有权，自动释放，零开销;
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    // 网络定义创建 Flag , 支持 0，1，2。
    NetworkDefinitionCreationFlags flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    // createNetworkV2 返回 INetworkDefinition* 裸指针，然后由智能指针接管
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flags));
    if (!network)
    {
        return false;
    }
    // 构建配置，指定 TensorRT 应该如何优化模型，通过配置其属性来实现，举例如下：
    // setMemoryPoolLimit 
    //        kWORKSPACE: 尽可能大的设置工作空间大小，期望以空间换取最佳性能；在单机多引擎场景，需要限制工作空间大小。
    //        kTACTIC_SHARED_MEMORY: 最大共享内存分配额度。在 TRT+DirectX 的多进程场景，至关重要。
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }
    // getPluginRegistry 返回全局插件注册表。
    // registerCreator 将 mPluginCreator注册到默认命名空间中（空字符串表示默认命名空间）
    // 若同类型插件已存在则返回 false，且调用是线程安全的（内部通过互斥锁同步）
    // REF：https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-api-migration.html?highlight=getPluginRegistry#using-plugin-shared-libraries
    bool const registered = getPluginRegistry()->registerCreator(mPluginCreator, "");
    ASSERT(registered && "Registration of NonZeroPluginCreator failed");
    
    // 创建一个包含 NonZeroZero 插件的单个自定义层网络，并标记输出层
    // build 指向将使用 NonZeroZero 插件进行填充的网络
    // builder 引擎构建器的指针 INetworkDefinition*
    auto constructed = constructNetwork(builder, network, config);
    if (!constructed)
    {
        return false;
    }
    
    // 单独创建一个 stream， 用于分析NN网络
    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    // b->b(): 返回IHostMemory*（指向序列化引擎数据的指针）
    // 当 plan离开作用域时，自动释放引擎数据
    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }
    // 创建 IRuntime 类的实例
    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime)
    {
        return false;
    }
    // 从主机内存中反序列化一个引擎 ICudaEngine
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 2);

    ASSERT(network->getNbOutputs() == 2);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 2);

    return true;
}

//!
//! \brief Creates a network with a single custom layer containing the NonZero plugin and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the NonZero plugin
//!
//! \param builder Pointer to the engine builder
//!
bool SampleNonZeroPlugin::constructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
    std::unique_ptr<nvinfer1::INetworkDefinition>& network, std::unique_ptr<nvinfer1::IBuilderConfig>& config)
{
    std::default_random_engine generator(mSeed);// default_random_engine 随机数引擎类型（模板类实例）,mseed 被按时间赋值了
    std::uniform_int_distribution<int32_t> distr(10, 25);// 均匀分布的整数分布器，闭区间 [10,25]
    // 随机的行数 R 和列数 C，作为输入层， 具有 2 个通道
    int32_t const R = distr(generator);
    int32_t const C = distr(generator);
    auto* in = network->addInput("Input", DataType::kFLOAT, {2, {R, C}});
    ASSERT(in != nullptr);

    if (mParams.fp16)
    {
        auto castLayer = network->addCast(*in, DataType::kHALF);
        ASSERT(castLayer != nullptr);
        in = castLayer->getOutput(0);
    }
    // 外层大括号：插件字段容器的初始化，内层大括号：插件的列表初始化
    std::vector<PluginField> const vecPF{{"rowOrder", &mParams.rowOrder, PluginFieldType::kINT32, 1}};
    // 初始化 PFC配置：字段数量, 字段数组指针
    PluginFieldCollection pfc{static_cast<int32_t>(vecPF.size()), vecPF.data()};
    
    // getCreator 获取插件创建器, 根据网络创建时与插件关联的插件名称、版本和命名空间
    // 存储为 IPluginCreatorV3One 对象
    auto pluginCreator = static_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator("NonZeroPlugin", "0", ""));
    ASSERT(pluginCreator != nullptr && "NonZeroPluginCreator not properly registered to registry");
    
    // 创建插件实例IPluginV3, 以名称，PFC配置，构建阶段 // createPlugin: 插件绑定入口
    auto plugin = std::unique_ptr<IPluginV3>(pluginCreator->createPlugin("NonZeroPlugin", &pfc, TensorRTPhase::kBUILD));
    ASSERT(plugin != nullptr && "NonZeroPlugin construction failed");

    // 现有的ITensor*指针类型变量 in，用于初始化 vector 的第一个元素
    std::vector<ITensor*> inputsVec{in};

    // 向网络增添 实现了 IPluginV3 接口的插件层
    auto pluginNonZeroLayer = network->addPluginV3(inputsVec.data(), inputsVec.size(), nullptr, 0, *plugin);
    ASSERT(pluginNonZeroLayer != nullptr);
    ASSERT(pluginNonZeroLayer->getOutput(0) != nullptr);
    ASSERT(pluginNonZeroLayer->getOutput(1) != nullptr);
    // 给张量一个可读的名称，便于后续通过名称获取输出。
    pluginNonZeroLayer->getOutput(0)->setName("Output0");
    pluginNonZeroLayer->getOutput(1)->setName("Output1");

    // 标记张量为网络输出:markOutput接受 ITensor&（引用，而非指针），必须解引用才能传递引用
    network->markOutput(*(pluginNonZeroLayer->getOutput(0)));
    network->markOutput(*(pluginNonZeroLayer->getOutput(1)));

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleNonZeroPlugin::infer()
{

    // 由于数据依赖的输出大小无法从引擎推断，因此需要为相应的输出缓冲区（以及其余的 I/O 张量）指定一个足够大的尺寸
    std::vector<int64_t> ioVolumes = {mInputDims.d[0] * mInputDims.d[1], mInputDims.d[0] * mInputDims.d[1] * 2, 1};

    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, ioVolumes);// TODO：概述功能

    // createExecutionContext： 创建一个执行上下文，并指定内部激活 memory 的分配策略。
    // 分配策略的默认值为 ExecutionContextAllocationStrategy::kSTATIC，这意味着上下文会预先分配一块足够满足所有 profile 文件的设备内存
    // 新创建的执行上下文将被分配optmize profile 0
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; ++i)
    {
        auto const name = mEngine->getIOTensorName(i);
        // setTensorAddress 为给定的输入或输出张量设置内存地址。
        // getDeviceBuffer 返回与 tensorName 对应的设备缓冲区。如果找不到该张量，则返回 nullptr。
        context->setTensorAddress(name, buffers.getDeviceBuffer(name));
    }

    // Read the input data into the managed buffers
    // 预处理输入数据并给到 buffer 管理，并打印 input
    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Create CUDA stream for the execution of this inference.
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

    bool status = context->enqueueV3(stream);
    if (!status)
    {
        return false;
    }

    // Asynchronously copy data from device output buffers to host output buffers.
    buffers.copyOutputToHostAsync(stream);

    // Wait for the work in the stream to complete.
    CHECK(cudaStreamSynchronize(stream));

    // Release stream.
    CHECK(cudaStreamDestroy(stream));

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleNonZeroPlugin::processInput(samplesCommon::BufferManager const& buffers)
{
    int32_t const inputH = mInputDims.d[0];
    int32_t const inputW = mInputDims.d[1];

    std::vector<uint8_t> fileData(inputH * inputW);

    // 0~9随机选取一个 手写数字的pgm 文件
    std::default_random_engine generator(mSeed);
    std::uniform_int_distribution<int32_t> distr(0, 9);
    auto const number = distr(generator);
    samplesCommon::readPGMFile(
        samplesCommon::locateFile(std::to_string(number) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    // 整形数据，读取为浮点数 f32 = 1 - u8/255
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int32_t i = 0; i < inputH * inputW; ++i)
    {
        auto const raw = 1.0 - float(fileData[i] / 255.0);
        hostDataBuffer[i] = raw;
    }

    sample::gLogInfo << "Input:" << std::endl;
    for (int32_t i = 0; i < inputH; ++i)
    {
        for (int32_t j = 0; j < inputW; ++j)
        {
            sample::gLogInfo << hostDataBuffer[i * inputW + j];
            if (j < inputW - 1)
            {
                sample::gLogInfo << ", ";
            }
        }
        sample::gLogInfo << std::endl;
    }
    sample::gLogInfo << std::endl;

    return true;
}

// 注释 NOLINTNEXTLINE 指示 Clang-Tidy 静态分析工具忽略下一行函数的认知复杂度警告，
// 表明开发者承认该函数逻辑复杂但设计合理，属于 TensorRT 等大型 C++ 项目中常见的代码质量控制手段。

//!
//! \brief Verify result
//!
//! \return whether the output correctly identifies all (and only) non-zero elements
//!
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool SampleNonZeroPlugin::verifyOutput(samplesCommon::BufferManager const& buffers)
{
    float* input = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    int32_t* output = static_cast<int32_t*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    int64_t count = *static_cast<int64_t*>(buffers.getHostBuffer(mParams.outputTensorNames[1]));

    std::vector<bool> covered(mInputDims.d[0] * mInputDims.d[1], false);

    sample::gLogInfo << "Output:" << std::endl;
    if (mParams.rowOrder)
    {
        for (int32_t i = 0; i < count; ++i)
        {
            for (int32_t j = 0; j < 2; ++j)
            {
                sample::gLogInfo << output[j + 2 * i] << " ";
            }
            sample::gLogInfo << std::endl;
        }
    }
    else
    {
        for (int32_t i = 0; i < 2; ++i)
        {
            for (int32_t j = 0; j < count; ++j)
            {
                sample::gLogInfo << output[j + count * i] << " ";
            }
            sample::gLogInfo << std::endl;
        }
    }

    if (!mParams.rowOrder)
    {
        for (int32_t i = 0; i < count; ++i)
        {
            auto const idx = output[i] * mInputDims.d[1] + output[i + count];
            covered[idx] = true;
            if (input[idx] == 0.F)
            {
                return false;
            }
        }
    }
    else
    {
        for (int32_t i = 0; i < count; ++i)
        {
            auto const idx = output[2 * i] * mInputDims.d[1] + output[2 * i + 1];
            covered[idx] = true;
            if (input[idx] == 0.F)
            {
                return false;
            }
        }
    }

    for (int32_t i = 0; i < static_cast<int32_t>(covered.size()); ++i)
    {
        if (!covered[i])
        {
            if (input[i] != 0.F)
            {
                return false;
            }
        }
    }

    return true;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
NonZeroParams initializeSampleParams(samplesCommon::Args const& args)
{
    NonZeroParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }

    params.inputTensorNames.push_back("Input");
    params.outputTensorNames.push_back("Output0");
    params.outputTensorNames.push_back("Output1");
    params.fp16 = args.runInFp16;
    params.rowOrder = args.rowOrder;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_non_zero_plugin [-h or --help] [-d or --datadir=<path to data directory>]"
              << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
    std::cout << "--columnOrder   Run plugin in column major output mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(kSAMPLE_NAME, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    SampleNonZeroPlugin sample(initializeSampleParams(args));// TODO: 解读

    sample::gLogInfo << "Building and running a GPU inference engine for NonZero plugin" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer()) 
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}

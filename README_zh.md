[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Documentation](https://img.shields.io/badge/TensorRT-documentation-brightgreen.svg)](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html) [![Roadmap](https://img.shields.io/badge/Roadmap-Q3_2026-brightgreen.svg)](documents/tensorrt_roadmap_2026q3.pdf)

 en [English](./README.md) ｜ zh_CN [简体中文](./README_zh.md)

# 📢📢📢 重要 Announcement 📢📢📢

TensorRT **11.0** 现已正式发布，带来一系列强大的新能力，专为加速你的 AI 推理工作流而设计。在这个主版本升级中，TensorRT 的 API 进行了精简整理，并移除了少量遗留特性。

下方列出了相关特性的迁移指南：

- **弱类型网络（Weakly-typed networks）及相关 API 已被移除**，由 https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/advanced.html#strongly-typed-networks 替代。
- **隐式量化（Implicit quantization）及相关 API 已被移除**，由 https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html#explicit-quantization 替代。
- **IPluginV2 及相关 API 已被移除**，由 https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html#migrating-v2-plugins-to-ipluginv3 替代。
- **TREX 工具已被移除**，由 https://docs.nvidia.com/nsight-dl-designer/UserGuide/index.html#visualizing-a-tensorrt-engine 替代。
- **Python 3.9 及更旧版本的 Python 绑定已被移除**。RHEL / Rocky Linux 8 与 RHEL / Rocky Linux 9 的 RPM 包现在依赖 **Python 3.12**。

---

# TensorRT 开源软件（OSS）

本仓库包含 **NVIDIA TensorRT 的开源软件组件（OSS）**，包括 TensorRT 插件与 ONNX 解析器的源码，以及演示 TensorRT 平台用法与能力的示例应用。这些开源组件是 TensorRT 正式通用版（GA）的一个子集，并附带一些扩展与 Bug Fix。

- 若想按步骤了解 TensorRT 各导入路径（ONNX、Torch-TensorRT、HuggingFace/Optimum、Network Definition API）并查看示例与工具技巧，请参阅 documents/import_workflows.md。
- 若想查看跨导入路径的逐模型支持矩阵（LLM、Encoder-NLP、视觉、音频、扩散模型、多模态），请参阅 documents/supported_models.md。
- 若希望向 TensorRT-OSS 贡献代码，请先阅读我们的 CONTRIBUTING.md 与 CODING-GUIDELINES.md。
- TensorRT-OSS 每次发布的新增内容与更新摘要，请查阅 CHANGELOG.md。
- 商业合作咨询请联系：mailto:researchinquiries@nvidia.com
- 媒体与其他咨询请联系 Hector Marinez：<a href="mailto:hmarinez@nvidia.com">hmarinez@nvidia.com</a>

需要企业级支持？TensorRT 可通过 https://www.nvidia.com/en-us/data-center/products/ai-enterprise/ 获得 NVIDIA 全球支持；并可前往 https://www.nvidia.com/en-us/launchpad/ai/ai-enterprise/ 免费体验一组托管在 NVIDIA 基础设施上的 TensorRT 动手实验环境。

加入 https://www.nvidia.com/en-us/deep-learning-ai/triton-tensorrt-newsletter/，第一时间获取产品更新、Bug Fix、技术内容、最佳实践等信息。

---

# 预编译 TensorRT Python 包

我们提供 TensorRT 的 Python 包，便于快速安装：

```bash
pip install tensorrt
```

如果你只用 Python 侧使用 TensorRT，可以跳过下方的 **构建（Build）** 章节。

---

# 构建（Build）

## 前置条件（Prerequisites）

要构建 TensorRT-OSS 组件，首先需要具备以下软件包。

### TensorRT GA 构建产物

- **TensorRT v11.0.0.114**
  - 可从下方直链下载

### 系统软件包（System Packages）

- [CUDA](https://developer.nvidia.com/cuda-toolkit)
  - Recommended versions:
  - cuda-13.2.0
  - cuda-12.9.0
- [CUDNN (optional)](https://developer.nvidia.com/cudnn)
  - cuDNN 8.9
- [GNU make](https://ftp.gnu.org/gnu/make/) >= v4.1
- [cmake](https://github.com/Kitware/CMake/releases) >= v3.31
- [python](https://www.python.org/downloads/) >= v3.10, <= v3.13.x
- [pip](https://pypi.org/project/pip/#history) >= v19.0
- Essential utilities
  - [git](https://git-scm.com/downloads), [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/), [wget](https://www.gnu.org/software/wget/faq.html#download)

### 可选软件包（Optional Packages）

- [NCCL](https://developer.nvidia.com/nccl/nccl-download) >= v2.19, < v3.0 — only when building with multi-device support (`-DTRT_BUILD_ENABLE_MULTIDEVICE=ON`) for the `sampleDistCollective` sample.
- Containerized build
  - [Docker](https://docs.docker.com/install/) >= 19.03
  - [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- PyPI packages (for demo applications/tests)
  - [onnx](https://pypi.org/project/onnx/)
  - [onnxruntime](https://pypi.org/project/onnxruntime/)
  - [tensorflow-gpu](https://pypi.org/project/tensorflow/) >= 2.5.1
  - [Pillow](https://pypi.org/project/Pillow/) >= 9.0.1
  - [pycuda](https://pypi.org/project/pycuda/) < 2021.1
  - [numpy](https://pypi.org/project/numpy/)
  - [pytest](https://pypi.org/project/pytest/)
- Code formatting tools (for contributors)

  - [Clang-format](https://clang.llvm.org/docs/ClangFormat.html)
  - [Git-clang-format](https://github.com/llvm-mirror/clang/blob/master/tools/clang-format/git-clang-format)

> **注意**：https://github.com/onnx/onnx-tensorrt、http://nvlabs.github.io/cub/、https://github.com/protocolbuffers/protobuf.git 会随 TensorRT OSS 一同下载，**无需手动安装**。

---

## 下载 TensorRT 构建产物

1. ### 克隆 TensorRT OSS 仓库

   ```bash
   git clone -b main https://github.com/nvidia/TensorRT TensorRT
   cd TensorRT
   git submodule update --init --recursive
   ```

2. ### （可选——若不使用 TensorRT 官方容器）指定 TensorRT GA Release 构建路径

   如果你使用的是 TensorRT OSS **构建专用容器**，TensorRT 库已预装到 `/usr/lib/x86_64-linux-gnu`，可跳过此步。

   否则，请从 https://developer.nvidia.com 下载并解压 TensorRT GA 构建产物（直链如下）：

   - https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/11.0.0/tars/TensorRT-Enterprise-11.0.0.114-Linux-x86_64-cuda-13.2-Release-external.tar.zst
   - https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/11.0.0/tars/TensorRT-Enterprise-11.0.0.114-Linux-x86_64-cuda-12.9-Release-external.tar.zst
   - https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/11.0.0/zip/TensorRT-Enterprise-11.0.0.114-Windows-amd64-cuda-13.2-Release-external.zip
   - https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/11.0.0/zip/TensorRT-Enterprise-11.0.0.114-Windows-amd64-cuda-12.9-Release-external.zip

   **示例：x86-64 Ubuntu 22.04 + cuda-13.2**

   ```bash
   cd ~/Downloads
   tar --zstd -xvf TensorRT-Enterprise-11.0.0.114-Linux-x86_64-cuda-13.2-Release-external.tar.zst
   export TRT_LIBPATH=`pwd`/TensorRT-11.0.0.114/lib
   ```

   **示例：x86-64 Windows + cuda-12.9**

   ```powershell
   Expand-Archive -Path TensorRT-Enterprise-11.0.0.114-Windows-amd64-cuda-12.9-Release-external.zip
   $env:TRT_LIBPATH="$pwd\TensorRT-11.0.0.114\lib"
   ```

---

## 搭建构建环境（Setting Up The Build Environment）

对于 Linux 平台，**建议使用 Docker 容器**来构建 TensorRT OSS（见下方流程）；若是原生构建（Native build），请先自行安装#prerequisites里的 **系统软件包**。

1. ### 生成 TensorRT-OSS 构建容器

   **示例：x86-64 Ubuntu 24.04 + cuda-13.2（默认）**

   ```bash
   ./docker/build.sh --file docker/ubuntu-24.04.Dockerfile --tag tensorrt-ubuntu24.04-cuda13.2
   ```

   **示例：x86-64 RockyLinux 8 + cuda-13.2**

   ```bash
   ./docker/build.sh --file docker/rockylinux8.Dockerfile --tag tensorrt-rockylinux8-cuda13.2
   ```

   **示例：Ubuntu 24.04 交叉编译 Jetson（aarch64）+ cuda-13.2（JetPack SDK）**

   ```bash
   ./docker/build.sh --file docker/ubuntu-cross-aarch64.Dockerfile --tag tensorrt-jetpack-cuda13.2
   ```

   **示例：aarch64 Ubuntu 24.04 原生构建 + cuda-13.2**

   ```bash
   ./docker/build.sh --file docker/ubuntu-24.04-aarch64.Dockerfile --tag tensorrt-aarch64-ubuntu24.04-cuda13.2
   ```

2. ### 启动 TensorRT-OSS 构建容器

   **示例：Ubuntu 24.04 构建容器**

   ```bash
   ./docker/launch.sh --tag tensorrt-ubuntu24.04-cuda13.2 --gpus all
   ```

   > **注意：**
   > 1. `--tag` 要与第 1 步生成的构建容器名对应。<br/>
   > 2. 构建容器内如需 GPU 访问（运行 TensorRT 应用），需要安装 #prerequisites。<br/>
   > 3. Ubuntu 构建容器的 `sudo` 密码为 `nvidia`。<br/>
   > 4. 可用 `--jupyter <port>` 指定端口以启动 Jupyter Notebook。<br/>
   > 5. 需要对当前文件夹拥有写权限（该目录会以 uid:gid=1000:1000 挂载进 Docker 容器内部）。

---

## 构建 TensorRT-OSS

- 生成 Makefiles 并构建

  **示例：Linux（x86-64）构建，默认 cuda-13.2**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out
  make -j$(nproc)
  ```

  **示例：Linux（aarch64）构建，默认 cuda-13.2**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64-native.toolchain
  make -j$(nproc)
  ```

  **示例：Jetson Thor（aarch64）原生构建 + cuda-13.2**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DTRT_PLATFORM_ID=aarch64
  CC=/usr/bin/gcc make -j$(nproc)
  ```

  > **注意**：aarch64 原生构建 protobuf 时，必须通过 `CC=` 显式指定 C 编译器。

  **示例：Ubuntu 24.04 交叉编译 Jetson Thor（aarch64）+ cuda-13.2（JetPack）**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64_cross.toolchain
  make -j$(nproc)
  ```

  **示例：Ubuntu 24.04 交叉编译 DriveOS（aarch64）+ cuda-13.2**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64_dos_cross.toolchain
  make -j$(nproc)
  ```

  **示例：Windows（x86）原生构建 + cuda-13.2**

  ```bash
  cd $TRT_OSSPATH
  New-Item -ItemType Directory -Path build
  cd build
  cmake .. -DTRT_LIB_DIR="$env:TRT_LIBPATH" -DTRT_OUT_DIR="$pwd\\out"
  msbuild TensorRT.sln /property:Configuration=Release -m:$env:NUMBER_OF_PROCESSORS
  ```

  > **注意**：CMake 使用的默认 CUDA 版本为 13.2；如需覆盖（例如改为 12.9），请在 cmake 命令后追加 `-DCUDA_VERSION=12.9`。

- ### 必需 CMake 构建参数

  - `TRT_LIB_DIR`：TensorRT 安装目录下包含库的 `lib` 路径。
  - `TRT_OUT_DIR`：构建产物输出目录（生成的构建制品会被拷贝到这里）。

- ### 可选 CMake 构建参数

  - `CMAKE_BUILD_TYPE`：指定生成 Release 还是 Debug（含调试符号）二进制文件，取值为 [`Release`] | `Debug`
  - `CUDA_VERSION`：目标 CUDA 版本，例如 [`12.9.9`]
  - `CUDNN_VERSION`：目标 cuDNN 版本，例如 [`8.9`]
  - `PROTOBUF_VERSION`：使用的 Protobuf 版本，例如 [`3.20.1`]。  
    > 注意：修改此项并不会让 CMake 改用系统 Protobuf，而是会让 CMake 下载并尝试构建指定版本。
  - `CMAKE_TOOLCHAIN_FILE`：交叉编译用的 toolchain 文件路径
  - `BUILD_PARSERS`：是否构建解析器，例如 [`ON`] | `OFF`。关闭后 CMake 会尝试在 `${TRT_LIB_DIR}` 和系统路径寻找预编译解析器库；Debug 构建时会优先选择 Debug 版本（若存在）。
  - `BUILD_PLUGINS`：是否构建插件，例如 [`ON`] | `OFF`。关闭后 CMake 会尝试在 `${TRT_LIB_DIR}` 和系统路径寻找预编译插件库；Debug 构建时同样优先选 Debug 版本。
  - `BUILD_SAMPLES`：是否构建示例，例如 [`ON`] | `OFF`
  - `BUILD_SAFE_SAMPLES`：是否构建安全（Safety）示例，例如 [`ON`] | `OFF`
  - `TRT_SAFETY_INFERENCE_ONLY`：是否只构建安全推理组件，例如 [`ON`] | `OFF`。开启后除 `BUILD_SAFE_SAMPLES` 外其他组件都会被关闭。
  - `TRT_PLATFORM_ID`：裸机构建（区别于容器化交叉编译），当前支持：`x86_64`（默认）。
  - `TRT_BUILD_ENABLE_MULTIDEVICE`：启用多设备示例（`sampleDistCollective`），通过 `-DTRT_BUILD_ENABLE_MULTIDEVICE=ON` 打开；需要 https://developer.nvidia.com/nccl/nccl-download >= v2.19，< v3.0。
  - `TRT_BUILD_TESTING`：为示例构建 gTests，需要 https://github.com/google/googletest；若不可用则在配置阶段自动拉取 googletest。

---

## 构建 TensorRT DriveOS 示例

- 生成 Makefiles 并构建

  **示例：交叉编译 DOS7 Linux（aarch64）**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DBUILD_SAMPLES=ON -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=OFF -DTRT_OUT_DIR=`pwd`/bin_dynamic_cross -DTRT_LIB_DIR=$TRT_LIBPATH -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64_dos_cross.toolchain
  make -j$(nproc)
  ```

  **示例：交叉编译 DOS6.5 Linux（aarch64）**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DBUILD_SAMPLES=ON -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=OFF -DTRT_OUT_DIR=`pwd`/bin_dynamic_cross -DTRT_LIB_DIR=$TRT_LIBPATH -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64_dos_cross.toolchain -DCUDA_VERSION=11.4 -DCMAKE_CUDA_ARCHITECTURES=87
  make -j$(nproc)
  ```

  **示例：DOS6.5 与 DOS7 Linux（aarch64）原生构建**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64-native.toolchain -DBUILD_SAMPLES=ON -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=OFF
  make -j$(nproc)
  ```

  **示例：交叉编译 DOS6.5 QNX（aarch64）**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  export CUDA_VERSION=11.4
  export CUDA=cuda-$CUDA_VERSION
  export CUDA_ROOT=/usr/local/cuda-safe-$CUDA_VERSION
  export QNX_BASE=/drive/toolchains/qnx_toolchain  # 修改为你的 QNX toolchain 安装路径
  export QNX_HOST=$QNX_BASE/host/linux/x86_64/
  export QNX_TARGET=$QNX_BASE/target/qnx7/
  export PATH=$PATH:$QNX_HOST/usr/bin
  cmake .. -DBUILD_SAMPLES=ON -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=OFF -DBUILD_SAFE_SAMPLES=OFF -DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc -DTRT_OUT_DIR=`pwd`/bin_dynamic_cross -DTRT_LIB_DIR=$TRT_LIBPATH -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_qnx.toolchain -DCUDA_VERSION=$CUDA_VERSION -DCMAKE_CUDA_ARCHITECTURES=87
  make -j$(nproc)
  ```

  > **注意**：请将 `QNX_BASE` 设置为你的 QNX toolchain 安装路径。  
  > 若你的 CUDA 版本与示例中不同，请设置 `CUDA_VERSION`（在多处用到它的示例里）或在 cmake 命令追加 `-DCUDA_VERSION=<version>`。

  **示例：交叉编译 DOS6.5 QNX Safety（aarch64）**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  export CUDA_VERSION=11.4
  export QNX_BASE=/drive/toolchains/qnx_toolchain  # 修改为你的 QNX toolchain 安装路径
  export QNX_HOST=$QNX_BASE/host/linux/x86_64/
  export QNX_TARGET=$QNX_BASE/target/qnx7/
  export PATH=$PATH:$QNX_HOST/usr/bin
  export CUDA=cuda-$CUDA_VERSION
  export CUDA_ROOT=/usr/local/cuda-safe-$CUDA_VERSION
  cmake .. -DBUILD_SAMPLES=OFF -DBUILD_SAFE_SAMPLES=ON -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=OFF -DTRT_SAFETY_INFERENCE_ONLY=ON -DTRT_OUT_DIR=`pwd`/bin_dynamic_cross -DTRT_LIB_DIR=$TRT_LIBPATH -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_qnx_safe.toolchain -DCUDA_VERSION=$CUDA_VERSION -DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=87
  make -j$(nproc)
  ```

  > **注意**：同上，`QNX_BASE` 需要指向你的 QNX toolchain 安装路径；CUDA 版本不一致时调整 `CUDA_VERSION` 或追加 `-DCUDA_VERSION=<version>`。

  **示例：交叉编译 DOS7 QNX（aarch64）**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  export CUDA_VERSION=13.2
  export CUDA=cuda-$CUDA_VERSION
  export CUDA_ROOT=/usr/local/cuda-safe-$CUDA_VERSION
  export QNX_BASE=/drive/toolchains/qnx_toolchain  # 修改为你的 QNX toolchain 安装路径
  export QNX_HOST=$QNX_BASE/host/linux/x86_64/
  export QNX_TARGET=$QNX_BASE/target/qnx/
  export PATH=$PATH:$QNX_HOST/usr/bin
  cmake .. -DBUILD_SAMPLES=ON -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=OFF -DBUILD_SAFE_SAMPLES=OFF -DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc -DTRT_OUT_DIR=`pwd`/bin_dynamic_cross -DTRT_LIB_DIR=$TRT_LIBPATH -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_qnx.toolchain -DCUDA_VERSION=$CUDA_VERSION -DCMAKE_CUDA_ARCHITECTURES=110
  make -j$(nproc)
  ```

  > **注意**：同上，`QNX_BASE` 需要指向你的 QNX toolchain 安装路径；CUDA 版本不一致时调整 `CUDA_VERSION` 或追加 `-DCUDA_VERSION=<version>`。

---

# 参考资料

## TensorRT 资源

- https://developer.nvidia.com/tensorrt
- https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html
- https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html
- https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html
- https://docs.nvidia.com/deeplearning/tensorrt/index.html#tools
- https://devtalk.nvidia.com/default/board/304/tensorrt/
- https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html

## 已知问题（Known Issues）

- 请参阅 https://docs.nvidia.com/deeplearning/tensorrt/release-notes
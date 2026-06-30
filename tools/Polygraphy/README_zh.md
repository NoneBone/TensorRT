

# Polygraphy：深度学习推理原型设计与调试工具包

## 目录

- [简介](#introduction)

- [安装](#installation)

- [命令行工具包](#command-line-toolkit)

- [Python API](#python-api)

- [示例](#examples)

- [操作指南](#how-to-guides)

- [贡献](#contributing)

## 简介

Polygraphy 是一款旨在协助在各种框架中运行和调试深度学习模型的工具包。它包括一个 [Python API](polygraphy)以及使用该 API 构建的 [命令行界面 (CLI)](polygraphy/tools)。

除此之外，Polygraphy 还支持以下功能：

- 在多个后端（如 TensorRT 和 ONNX-Runtime）之间运行推理并比较结果

  （例如：[API](examples/api/01_comparing_frameworks/)、[CLI](examples/cli/run/01_comparing_frameworks/)）

- 将模型转换为多种格式，例如带有训练后量化的 TensorRT 引擎

  （例如：[API](examples/api/04_int8_calibration_in_tensorrt/)、[CLI](examples/cli/convert/01_int8_calibration_in_tensorrt/)）

- 查看各类模型的信息

  （例如：[CLI](examples/cli/inspect/)）

- 在命令行中修改 ONNX 模型：

  -   提取子图（例如：[CLI](examples/cli/surgeon/01_isolating_subgraphs/)）

  -   简化与清理（例如：[CLI](examples/cli/surgeon/02_folding_constants/)）

- 隔离 TensorRT 中存在问题的策略（tactic）

  （例如：[CLI](examples/cli/debug/01_debugging_flaky_trt_tactics/)）

## 安装

**重要提示**：**Polygraphy 仅支持 Python 3.6 及更高版本。**

**在按照以下说明操作之前，请确保您使用的是受支持的 Python 版本。**

### 安装预构建的 Wheel 文件

```
python -m pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com
```

**注意：** *在 Linux 上，命令行工具包通常默认安装到 `${HOME}/.local/bin`。*

*请确保将此目录添加到您的 `PATH`环境变量中。*

### 从源码构建

#### 使用 Make 目标（Linux）

```
make install
```

#### 使用 PowerShell 脚本（Windows）

请确保您的系统允许执行脚本，然后运行：

```
.\install.ps1
```

#### 手动构建

1. 安装前置依赖：

```
python -m pip install wheel
```

1. 构建一个 wheel 文件：

```
python setup.py bdist_wheel
```

1. 在仓库**外部**手动安装 wheel 文件：

   在 Linux 上，运行：

   ```
   python -m pip install Polygraphy/dist/polygraphy-*-py2.py3-none-any.whl
   ```

   在 Windows 上，使用 PowerShell 运行：

   ```
   $wheel_path = gci -Name Polygraphy\dist
   python -m pip install Polygraphy\dist\$wheel_path
   ```

   **注意：** *强烈建议安装 `colored`模块以获得彩色输出，*

   *这可以极大地提高可读性：*

   ```
   python -m pip install colored
   ```

### 安装依赖项

Polygraphy 对其他 Python 包没有硬性依赖。但是，其中包含的大部分功能确实需要其他 Python 包的支持。

#### 自动安装依赖项

由于具体需要哪些包取决于实际使用的功能，因此很难提前确定所有必需的包。

为了简化这一过程，Polygraphy 可以选择在运行时根据需要自动安装或升级依赖项。要启用此行为，请将 `POLYGRAPHY_AUTOINSTALL_DEPS`环境变量设置为 `1`，或使用 Python API 设置 `polygraphy.config.AUTOINSTALL_DEPS = True`。

**注意**：*默认情况下，依赖项将使用当前解释器进行安装，并可能会覆盖现有包。*

*可以通过设置 `POLYGRAPHY_INSTALL_CMD`环境变量，或使用 Python API 设置 `polygraphy.config.INSTALL_CMD`来覆盖默认的安装命令（即 `python -m pip install`）。*

如果您希望在自动安装或升级包之前收到 Polygraphy 的提示，请将 `POLYGRAPHY_ASK_BEFORE_INSTALL`环境变量设置为 `1`，或使用 Python API 设置 `polygraphy.config.ASK_BEFORE_INSTALL = True`。

#### 手动安装

每个 `backend`目录都包含一个 `requirements.txt`文件，其中指定了该后端所依赖的最低限度的包集合。这并不一定包含该后端提供的所有功能所需的所有包，但可以作为良好的起点。

您可以为您感兴趣的后端安装依赖项：

```
python -m pip install -r polygraphy/backend/<name>/requirements.txt
```

如果需要额外的包，将会记录警告或错误信息。您可以手动安装这些额外的包：

```
python -m pip install <package_name>
```

## 命令行工具包

有关 Polygraphy 工具包中包含的各种工具的详细信息，请参阅 [CLI 用户指南](polygraphy/tools)。

### Python API

有关 Polygraphy Python API 的更多信息，包括高层概述和 Python API 参考文档，请参阅 [API 目录](polygraphy)。

## 示例

有关 CLI 和 Python API 的示例，请参阅 [示例目录](examples)。

## 操作指南

有关操作指南，请参阅 [操作指南目录](how-to)。

## 贡献

如需了解如何为本项目做出贡献，请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。
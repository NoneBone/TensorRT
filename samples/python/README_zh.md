# 示例通用设置指南
==============================

 en [English](./README.md) ｜ zh_CN [简体中文](./README_zh.md)


## 下载示例数据

通过 `python3 -m pip install -r requirements.txt` 安装工具依赖项。

如果示例目录中存在 `download.yml`（onnx_packnet/download.yml），请调用 downloader.py 下载数据，命令如下：

```sh
downloader.py -d /path/to/data/dir -f /path/to/download.yml
python downloader.py -d /root/cys/PROJECT/00-COMMON/DEMO/02-TensorRT/data/ -f /root/cys/PROJECT/00-COMMON/DEMO/02-TensorRT/samples/python/onnx_packnet/download.yml

```

数据目录（即 `/path/to/data/dir`）是一个用于存储所有示例数据的集中式目录。因此，您可以为所有示例使用同一个目录。可以通过 `-d /path/to/data/dir` 或环境变量 `$TRT_DATA_DIR` 提供该路径，其中 `-d` 参数的优先级更高。

在运行依赖已下载数据的示例脚本时，请务必使用 `-d` 或 `$TRT_DATA_DIR`。如果在数据目录中未找到已下载的数据，脚本将中止运行（使用 `$TRT_DATA_DIR` 会更加简便）。如果数据未正确设置，将会抛出错误。

`download.yml` 文件由示例自身维护，其中描述了示例名称、路径、URL 以及该示例所需数据文件的校验和。


**面向示例开发者的注意事项**

若要使用已下载的数据文件，请将类似以下的代码段集成到示例代码中，并通过传入该示例关联的 `download.yml` 文件中指定的 `path` 来获取数据文件的路径。

```py
TRT_DATA_DIR = None

def getFilePath(path):
    global TRT_DATA_DIR
    if not TRT_DATA_DIR:
        parser = argparse.ArgumentParser(description="将 PackNet 转换为 ONNX")
        parser.add_argument('-d', '--data', help="指定数据保存的目录。此参数将覆盖 $TRT_DATA_DIR。")
        args, _ = parser.parse_known_args()
        TRT_DATA_DIR = os.environ.get('TRT_DATA_DIR', None) if args.data is None else args.data
    if TRT_DATA_DIR is None:
        raise ValueError("必须通过 '-d $DATA' 或环境变量 $TRT_DATA_DIR 指定数据目录。")

    fullpath = os.path.join(TRT_DATA_DIR, path)
    if not os.path.exists(fullpath):
        raise ValueError("数据文件 %s 不存在！" % fullpath)

    return fullpath
```

**Python 版本支持**

所有 Python 示例均需在 Python>=3.10 环境下运行。不建议使用任何更低版本，因为这可能会导致兼容性问题。

# 更新日志

2025 年 8 月
移除了对 Python < 3.10 版本的支持。
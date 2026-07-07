# 介绍

此演示应用程序（“demoDiffusion”）展示了使用 TensorRT 加速稳定扩散和 ControlNet 管道。

# 设置

### 克隆 TensorRT OSS 代码库

```bash
git clone git@github.com:NVIDIA/TensorRT.git -b release/11.0 --single-branch
cd TensorRT
```

### 启动 NVIDIA PyTorch 容器

使用以下命令安装 nvidia-docker [these intructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)。

```bash
# Create a directory for persistent dependencies
mkdir -p deps

# Launch container with volume mounts
docker run --rm -it --gpus all \
  -v $PWD:/workspace \
  -v $PWD/deps:/workspace/deps \
  nvcr.io/nvidia/pytorch:26.03-py3 /bin/bash
```

> **注意：** 安装 `/workspace/deps` 作为卷，可以确保依赖项在容器重启后仍然存在。初始安装后，后续容器启动将重用已安装的依赖项。

注意：此演示支持 CUDA 13.0 及更高版本。

### 安装所需的软件包

此演示使用基于族的依赖关系管理系统。请安装您要使用的模型族的依赖项：

**安装所有依赖项（建议首次使用的用户执行此操作）：**
```bash
python3 setup.py all
```

或者安装特定的模型系列：
```bash
# SD family: SD 1.4, SDXL, SD3, SD3.5, SVD (Stable Video Diffusion), Stable Cascade
python3 setup.py sd

# Flux family: Flux.1-dev, Flux.1-schnell, Flux.1-Canny, Flux.1-Depth, Flux.1-Kontext
python3 setup.py flux

# Cosmos family: Cosmos-Predict2 text2image, video2world
python3 setup.py cosmos
```

**其他选项：**
```bash
# Force reinstall even if already installed
python3 setup.py all --force

# Install dependencies to a custom location
# Option 1 (recommended): set the env var and install
export TENSORRT_DIFFUSION_DEPS_ROOT=/custom/path/deps
python3 setup.py all

# Option 2: install to a path without changing this shell
# Remember to export the env var in the environment that runs the demos,
# so deps.configure() can find the custom path.
python3 setup.py all --deps-root /custom/path/deps
# Then, before running any demo scripts:
export TENSORRT_DIFFUSION_DEPS_ROOT=/custom/path/deps
```

**检查安装状态：**
```bash
python3 -c "from demo_diffusion import deps; deps.print_status()"
```

使用以下命令检查已安装的 TensorRT 版本：
```bash
python3 -c 'import tensorrt; print(tensorrt.__version__)'
```

> 注意：或者，您可以从以下位置下载并安装 TensorRT 软件包： [NVIDIA TensorRT Developer Zone](https://developer.nvidia.com/tensorrt)。

> 注意：demoDiffusion 已在配备 NVIDIA H100、A100、L40、T4 和 RTX4090 GPU 的系统上进行了测试，并采用了以下软件配置。


# 运行演示扩散

### 查看所支持管道的使用说明

```bash
python3 demo_txt2img.py --help
python3 demo_img2img.py --help
python3 demo_controlnet.py --help
python3 demo_txt2img_xl.py --help
python3 demo_txt2img_flux.py --help
python3 demo_txt2vid_wan.py --help
```

### HuggingFace 用户访问令牌

要下载稳定扩散管道的模型检查点，请获取 `read` HuggingFace Hub 的访问令牌。请参阅 [instructions](https://huggingface.co/docs/hub/security-tokens)。

```bash
export HF_TOKEN=<your access token>
```

### 根据文本提示生成图像

```bash
python3 demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN
```

### 使用基于模型优化的 SD1.4 INT8 和 FP8 量化实现更快的文本到图像转换

运行以下命令，生成一个 INT8 模式下 SD1.4 的图像。

```bash
python3 demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --int8
```

运行以下命令以生成采用 SD1.4 和 FP8 格式的图像。（FP8 仅 Hopper 和 Ada 平台支持。）

```bash
python3 demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --fp8
```

### 根据初始图像和文本提示生成图像

```bash
wget https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg -O sketch-mountains-input.jpg

python3 demo_img2img.py "A fantasy landscape, trending on artstation" --hf-token=$HF_TOKEN --input-image=sketch-mountains-input.jpg
```

### 使用 ControlNet，根据图像和文本提示生成图像

```bash
python3 demo_controlnet.py "Stormtrooper's lecture in beautiful lecture hall" --controlnet-type depth --hf-token=$HF_TOKEN --denoising-steps 20 --onnx-dir=onnx-cnet-depth --engine-dir=engine-cnet-depth
```

> 笔记： `--input-image` 必须是与以下图像对应的预处理图像： `--controlnet-type`如果未指定，则会下载示例图像。支持的控制网类型包括： `canny`， `depth`， `hed`， `mlsd`， `normal`， `openpose`， `scribble`， 和 `seg`。

例如：
<img src="https://drive.google.com/uc?export=view&id=17ub3MVSQHp26ty-wioNX6iQQ-nAveYSV" alt= “” width="800" height="400">

#### 结合多种条件反射

还可以指定多种 ControlNet 类型来组合条件。指定多个条件时，还应提供 ControlNet 标度。标度表示每个条件相对于其他条件的重要性。例如，要使用以下条件进行控制： `openpose` 和 `canny` 分别以 1.0 和 0.8 为尺度，所提供的论点将是： `--controlnet-type openpose canny` 和 `--controlnet-scale 1.0 0.8`请注意，提供的控制网尺度数量应与控制网类型数量相匹配。

### 使用 Stable Diffusion XL，根据单个文本提示生成图像

> **注意：** SDXL 及更高版本的稳定扩散模型需要安装 sd 依赖项。安装方法： `python3 setup.py sd`

运行以下命令以使用 Stable Diffusion XL 生成图像

```bash
python3 demo_txt2img_xl.py "a photo of an astronaut riding a horse on mars" --hf-token=$HF_TOKEN --version=xl-1.0
```

可通过指定以下参数启用可选的精炼器模型： `--enable-refiner` 使用单独的目录来存储精炼器 ONNX 文件和引擎文件 `--onnx-refiner-dir` 和 `--engine-refiner-dir` 分别。

```bash
python3 demo_txt2img_xl.py "a photo of an astronaut riding a horse on mars" --hf-token=$HF_TOKEN --version=xl-1.0 --enable-refiner --onnx-refiner-dir=onnx-refiner --engine-refiner-dir=engine-refiner
```

### 使用 ControlNet 和 Stable Diffusion XL，根据图像和文本提示生成图像

```bash
python3 demo_controlnet.py "A beautiful bird with rainbow colors" --controlnet-type canny --hf-token=$HF_TOKEN --denoising-steps 20 --onnx-dir=onnx-cnet --engine-dir=engine-cnet --version xl-1.0
```

> 注意：目前仅限 `--controlnet-type canny` 已支持。 `--input-image` 必须是与以下图像对应的预处理图像： `--controlnet-type canny`如果未指定，则会下载示例图像。

> 注：FP8 量化（`--fp8`) 受支持。

### 根据文本提示，并使用指定的 LoRa 模型权重更新，生成图像。

```bash
# FP16
python3 demo_txt2img_xl.py "Picture of a rustic Italian village with Olive trees and mountains" --version=xl-1.0 --lora-path "ostris/crayon_style_lora_sdxl" "ostris/watercolor_style_lora_sdxl" --lora-weight 0.3 0.7 --onnx-dir onnx-sdxl-lora --engine-dir engine-sdxl-lora --build-enable-refit

# FP8
python3 demo_txt2img_xl.py "Picture of a rustic Italian village with Olive trees and mountains" --version=xl-1.0 --lora-path "ostris/crayon_style_lora_sdxl" "ostris/watercolor_style_lora_sdxl" --lora-weight 0.3 0.7 --onnx-dir onnx-sdxl-lora --engine-dir engine-sdxl-lora --fp8
```

### 使用 ModelOpt 实现基于 SDXL INT8 和 FP8 量化的更快文本到图像转换

运行以下命令，以 INT8 格式生成使用 Stable Diffusion XL 的图像。

```bash
python3 demo_txt2img_xl.py "a photo of an astronaut riding a horse on mars" --version xl-1.0 --onnx-dir onnx-sdxl --engine-dir engine-sdxl --int8
```

运行以下命令，以 FP8 格式生成使用 Stable Diffusion XL 的图像。（FP8 格式仅支持 Hopper 和 Ada 平台。）

```bash
python3 demo_txt2img_xl.py "a photo of an astronaut riding a horse on mars" --version xl-1.0 --onnx-dir onnx-sdxl --engine-dir engine-sdxl --fp8
```

> 请注意，INT8 和 FP8 量化仅 SDXL 支持，不适用于 LoRA 权重。FP8 量化仅 Hopper 和 Ada 支持。某些提示符可能需要较少的去噪步骤才能产生更好的输入（例如）。 `--denoising-steps 20`但这将重复 U-Net 的校准、ONNX 导出和引擎构建过程。

有关在稳定扩散模型上运行 INT8 和 FP8 推理的分步教程，请参阅示例。 [TensorRT ModelOpt diffusers sample](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/diffusers)。

### 使用 SDXL Turbo 实现更快的文本转图像

只需一步即可生成连贯的图像。注意：SDXL Turbo 在 512x512 分辨率、禁用 EulerA 调度器和无分类器引导的情况下效果最佳。

```bash
python3 demo_txt2img_xl.py "Einstein" --version xl-turbo --onnx-dir onnx-sdxl-turbo --engine-dir engine-sdxl-turbo --denoising-steps 1 --scheduler EulerA --guidance-scale 0.0 --width 512 --height 512
```

### 使用稳定扩散 3 及其变体，根据文本提示生成图像

运行以下命令，使用稳定扩散 3 和稳定扩散 3.5 生成图像。

```bash
# Stable Diffusion 3
python3 demo_txt2img_sd3.py "A vibrant street wall covered in colorful graffiti, the centerpiece spells \"SD3 MEDIUM\", in a storm of colors" --version sd3 --hf-token=$HF_TOKEN

# Stable Diffusion 3.5-medium
python3 demo_txt2img_sd35.py "a beautiful photograph of Mt. Fuji during cherry blossom" --version=3.5-medium --denoising-steps=30 --guidance-scale 3.5 --hf-token=$HF_TOKEN --bf16 --download-onnx-models

# Stable Diffusion 3.5-large
python3 demo_txt2img_sd35.py "a beautiful photograph of Mt. Fuji during cherry blossom" --version=3.5-large --denoising-steps=30 --guidance-scale 3.5 --hf-token=$HF_TOKEN --bf16 --download-onnx-models

# Stable Diffusion 3.5-large FP8
python3 demo_txt2img_sd35.py "a beautiful photograph of Mt. Fuji during cherry blossom" --version=3.5-large --denoising-steps=30 --guidance-scale 3.5 --hf-token=$HF_TOKEN --fp8 --download-onnx-models --onnx-dir onnx_35_fp8/ --engine-dir engine_35_fp8/
```

您还可以指定输入图像条件，如下所示。

```bash
wget https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png -O dog-on-bench.png

# Stable Diffusion 3
python3 demo_txt2img_sd3.py "dog wearing a sweater and a blue collar" --version sd3 --input-image dog-on-bench.png --hf-token=$HF_TOKEN
```

请注意，当提供输入图像条件时，去噪百分比会影响去噪步骤数。其默认值为 0.6。可以使用以下命令更新此参数。 `--denoising-percentage`

### 使用 ControlNet 和 Stable Diffusion v3.5-large，根据图像和文本提示生成图像

```bash
# Depth BF16
python3 demo_controlnet_sd35.py "a photo of a man" --controlnet-type depth --hf-token=$HF_TOKEN --denoising-steps 40 --guidance-scale 4.5 --bf16 --download-onnx-models --low-vram

# Depth FP8
python3 demo_controlnet_sd35.py "a photo of a man" --version=3.5-large --fp8 --controlnet-type depth --download-onnx-models --denoising-steps=40 --guidance-scale 4.5 --hf-token=$HF_TOKEN --low-vram

# Canny BF16
python3 demo_controlnet_sd35.py "A Night time photo taken by Leica M11, portrait of a Japanese woman in a kimono, looking at the camera, Cherry blossoms" --controlnet-type canny --hf-token=$HF_TOKEN --denoising-steps 60 --guidance-scale 3.5 --bf16 --download-onnx-models --low-vram

# Canny FP8
python3 demo_controlnet_sd35.py "A Night time photo taken by Leica M11, portrait of a Japanese woman in a kimono, looking at the camera, Cherry blossoms" --version=3.5-large --fp8 --controlnet-type canny --hf-token=$HF_TOKEN --denoising-steps 60 --guidance-scale 3.5 --low-vram --download-onnx-models

# Blur
python3 demo_controlnet_sd35.py "generated ai art, a tiny, lost rubber ducky in an action shot close-up, surfing the humongous waves, inside the tube, in the style of Kelly Slater" --controlnet-type blur --hf-token=$HF_TOKEN --denoising-steps 60 --guidance-scale 3.5 --bf16 --download-onnx-models --low-vram
```

### 使用稳定视频扩散技术，根据初始图像生成视频。

下载预导出的 ONNX 模型

```bash
pip install -U "huggingface_hub[cli]"
hf download stabilityai/stable-video-diffusion-img2vid-xt-1-1-tensorrt --local-dir onnx-svd-xt-1-1
```

SVD-XT-1.1（25帧，分辨率576x1024）

```bash
python3 demo_img2vid.py --version svd-xt-1.1 --onnx-dir onnx-svd-xt-1-1 --engine-dir engine-svd-xt-1-1 --hf-token=$HF_TOKEN
```

运行以下命令生成 FP8 格式的视频。

```bash
python3 demo_img2vid.py --version svd-xt-1.1 --onnx-dir onnx-svd-xt-1-1 --engine-dir engine-svd-xt-1-1 --hf-token=$HF_TOKEN --fp8
```

> 注意：HuggingFace 存在一个 bug，您可以按照以下步骤进行解决方法。 [PR](https://github.com/huggingface/diffusers/pull/6562/files)

```
if torch.is_tensor(num_frames):
    num_frames = num_frames.item()
emb = emb.repeat_interleave(num_frames, dim=0)
```

您还可以使用以下方式指定自定义条件映像 `--input-image`：

```bash
python3 demo_img2vid.py --version svd-xt-1.1 --onnx-dir onnx-svd-xt-1-1 --engine-dir engine-svd-xt-1-1 --input-image https://www.hdcarwallpapers.com/walls/2018_chevrolet_camaro_zl1_nascar_race_car_2-HD.jpg --hf-token=$HF_TOKEN
```

注意：最小和最大引导比例尺分别使用 --min-guidance-scale 和 --max-guidance-scale 进行配置。

### 使用稳定级联算法，根据文本提示生成图像。

运行以下命令，使用稳定级联生成图像。

```bash
python3 demo_stable_cascade.py --onnx-opset=16 "Anthropomorphic cat dressed as a pilot" --onnx-dir onnx-sc --engine-dir engine-sc
```

以下命令也支持模型的精简版。

```bash
python3 demo_stable_cascade.py --onnx-opset=16 "Anthropomorphic cat dressed as a pilot" --onnx-dir onnx-sc-lite --engine-dir engine-sc-lite --lite
```

> 注意：该流程仅适用于 BF16 模型权重。

> 注意：该管道仅支持使用 Opset 16 导出 ONNX 文件。

> 注意：先验模型和解码器模型的去噪步骤和指导尺度分别使用 --prior-denoising-steps、--prior-guidance-scale、--decoder-denoising-steps 和 --decoder-guidance-scale 进行配置。

### 使用 Flux 生成图像

> **注意：** Flux 模型需要 Flux 系列依赖项。安装方法： `python3 setup.py flux`

#### 1. 根据文本提示生成图像

##### 运行 Flux.1-Dev

注意：通过 `--download-onnx-models` 为了避免原生 ONNX 导出，并从以下位置下载 ONNX 模型： [Black Forest Labs' collection](https://huggingface.co/collections/black-forest-labs/flux1-onnx-679d06b7579583bd84c8ef83)它仅支持 BF16、FP8 和 FP4 流水线。

```bash
# FP16 (requires >48GB VRAM for native export)
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN

# BF16
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --bf16 --download-onnx-models

# FP8
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --quantization-level 4 --fp8 --download-onnx-models

# FP4
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --fp4 --download-onnx-models
```

##### 运行 Flux.1-快速

```bash
# FP16 (requires >48GB VRAM for native export)
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --version="flux.1-schnell"

# BF16
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --version="flux.1-schnell" --bf16 --download-onnx-models

# FP8
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --version="flux.1-schnell" --quantization-level 4 --fp8 --download-onnx-models

# FP4
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --version="flux.1-schnell" --fp4 --download-onnx-models
```

---

#### 2. 根据初始图像和文本提示生成图像

下载示例输入图像：

```bash
wget "https://miro.medium.com/v2/resize:fit:640/format:webp/1*iD8mUonHMgnlP0qrSx3qPg.png" -O yellow.png
```

运行图像到图像的处理流程：

```bash
python3 demo_img2img_flux.py "A home with 2 floors and windows. The front door is purple" --hf-token=$HF_TOKEN --input-image yellow.png --image-strength 0.95 --bf16 --onnx-dir onnx-flux-dev/bf16 --engine-dir engine-flux-dev/
```

---

#### 3. 使用 Flux ControlNet 生成图像

##### 下载控制图像

```bash
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png
```

##### 用于原生 ONNX 导出（FP8 流水线）的校准数据

FP8 ControlNet流程需要下载校准数据集并提供其路径。您可以使用Black Forest Labs提供的数据集，链接如下： [depth](https://drive.google.com/file/d/1DFfhOSrTlKfvBFLcD2vAALwwH4jSGdGk/view) | [canny](https://drive.google.com/file/d/1dRoxOL-vy3tSAesyqBSJoUWsbkMwv3en/view)

你可以使用 `--calibraton-dataset` 用于指定路径的标志，设置为 `./{depth/canny}-eval/benchmark` 如果未提供，则使用默认值。请注意，数据集应包含 `inputs/` 和 `prompts/` 在提供的路径下方，与 BFL 数据集的格式相匹配。

##### 深度控制网

```bash
# BF16
python3 demo_img2img_flux.py "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts." --version="flux.1-dev-depth" --hf-token=$HF_TOKEN --guidance-scale 10 --control-image robot.png --bf16 --denoising-steps 30  --download-onnx-models

# FP8 using pre-exported ONNX models
python3 demo_img2img_flux.py "A robot made of exotic candies" --version="flux.1-dev-depth" --hf-token=$HF_TOKEN --guidance-scale 10 --control-image robot.png --fp8 --denoising-steps 30 --download-onnx-models --build-static-batch --quantization-level 4

# FP8 using native ONNX export
rm -rf onnx/* engine/* && python3 demo_img2img_flux.py "A robot made of exotic candies" --version="flux.1-dev-depth" --hf-token=$HF_TOKEN --guidance-scale 10 --control-image robot.png --quantization-level 4 --fp8 --denoising-steps 30

# FP4
python3 demo_img2img_flux.py "A robot made of exotic candies" --version="flux.1-dev-depth" --hf-token=$HF_TOKEN --guidance-scale 10 --control-image robot.png --fp4 --denoising-steps 30 --download-onnx-models --build-static-batch
```

##### Canny ControlNet

```bash
# BF16
python3 demo_img2img_flux.py "a robot made out of gold" --version="flux.1-dev-canny" --hf-token=$HF_TOKEN --guidance-scale 30 --control-image robot.png --bf16 --denoising-steps 30 --download-onnx-models

# FP8 using pre-exported ONNX models
python3 demo_img2img_flux.py "a robot made out of gold" --version="flux.1-dev-canny" --hf-token=$HF_TOKEN --guidance-scale 30 --control-image robot.png --fp8 --denoising-steps 30 --download-onnx-models --build-static-batch --quantization-level 4

# FP8 using native ONNX export
rm -rf onnx/* engine/* && python3 demo_img2img_flux.py "a robot made out of gold" --version="flux.1-dev-canny" --hf-token=$HF_TOKEN --guidance-scale 30 --control-image robot.png --quantization-level 4 --fp8 --denoising-steps 30 --calibration-dataset {custom/dataset/path}

# FP4
python3 demo_img2img_flux.py "a robot made out of gold" --version="flux.1-dev-canny" --hf-token=$HF_TOKEN --guidance-scale 30 --control-image robot.png --fp4 --denoising-steps 30 --download-onnx-models --build-static-batch
```

#### 4. 使用 Flux LoRA 生成图像

FLUX 支持为 Flux.1-Dev 和 Flux.1-Schnell 加载 LoRa。请确保目标 LoRa 与 Transformer 模型兼容。以下是一个使用示例： [water color Flux LoRA](https://huggingface.co/SebastianBodza/flux_lora_aquarel_watercolor)

```bash
# FP16
python3 demo_txt2img_flux.py "A painting of a barista creating an intricate latte art design, with the 'Coffee Creations' logo skillfully formed within the latte foam. In a watercolor style, AQUACOLTOK. White background." --hf-token=$HF_TOKEN --lora-path "SebastianBodza/flux_lora_aquarel_watercolor" --lora-weight 1.0 --onnx-dir=onnx-flux-lora --engine-dir=engine-flux-lora

# FP8
python3 demo_txt2img_flux.py "A painting of a barista creating an intricate latte art design, with the 'Coffee Creations' logo skillfully formed within the latte foam. In a watercolor style, AQUACOLTOK. White background." --hf-token=$HF_TOKEN --lora-path "SebastianBodza/flux_lora_aquarel_watercolor" --lora-weight 1.0 --onnx-dir=onnx-flux-lora --engine-dir=engine-flux-lora --fp8
```

#### 5. 使用 Flux Kontext 编辑图像

```bash
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png

# BF16
python3 demo_img2img_flux.py "Add a hat to the cat" --version="flux.1-kontext-dev" --hf-token=$HF_TOKEN --guidance-scale 2.5 --kontext-image cat.png --denoising-steps 28 --bf16 --onnx-dir onnx-kontext --engine-dir engine-kontext --download-onnx-models

# FP8
python3 demo_img2img_flux.py "Add a hat to the cat" --version="flux.1-kontext-dev" --hf-token=$HF_TOKEN --guidance-scale 2.5 --kontext-image cat.png --denoising-steps 28 --fp8 --onnx-dir onnx-kontext-fp8 --engine-dir engine-kontext-fp8 --download-onnx-models --quantization-level 4

# FP4
python3 demo_img2img_flux.py "Add a hat to the cat" --version="flux.1-kontext-dev" --hf-token=$HF_TOKEN --guidance-scale 2.5 --kontext-image cat.png --denoising-steps 28 --fp4 --onnx-dir onnx-kontext-fp4 --engine-dir engine-kontext-fp4 --download-onnx-models
```
---

#### 5. 仅导出 ONNX 模型（跳过推理）

使用 `--onnx-export-only` 此标志用于将 ONNX 模型导出到具有更高显存 (VRAM) 的设备。导出的 ONNX 模型可用于显存较低的设备，以进行引擎构建和推理步骤。

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --onnx-export-only
```

---

#### 6. 在显存有限的 GPU 上运行 Flux

##### 优化标志

- `--low-vram`：启用模型卸载功能，以减少 VRAM 使用量。
- `--ws`：在 TensorRT 引擎中启用权重流。
- `--t5-ws-percentage` 和 `--transformer-ws-percentage`设置运行时权重流预算。
- `--build-static-batch`：使用静态批处理大小构建所有引擎，以降低所需的激活内存。这将限制这些引擎推理支持的批处理大小为指定的值。 `--batch-size`。

##### FLUX VRAM 需求表

下面记录的内存使用情况不包括 ONNX 导出步骤，并假设使用了 `--build-static-batch` 启用此标志可减少激活时的显存占用。用户可以选择使用 [pre-exported ONNX models](README.md#download-pre-exported-models-recommended-for-48gb-vram) 或者使用以下方式将模型单独导出到具有更高显存容量的设备上： [--onnx-export-only](README.md#4-export-onnx-models-only-skip-inference)。

| 精度 | 默认显存使用率 | 使用 `--low-vram` |
| --------- | ------------------ | ----------------- |
| FP16 | 39.3GB | 23.9GB |
| BF16 | 35.7 GB | 23.9 GB |
| FP8 | 24.6GB | 14.9GB |
| FP4 | 21.67 GB | 11.1 GB |

注意：FP8 和 FP4 流水线仅在 Hopper/Ada/Blackwell 设备上受支持。FP4 流水线在 Blackwell 设备上的性能最佳。


### 运行 Cosmos2 世界基金会模型

> **注意：** Cosmos 模型需要 Cosmos 系列依赖项。请使用以下命令安装： `python3 setup.py cosmos`

选择提示并将其导出，如下所示。

```bash
export PROMPT="A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess."

export NEGATIVE_PROMPT="The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
```

#### 1. 根据文本提示生成图像

##### 运行 Cosmos-Predict2-2B-Text2Image

```bash
# BF16
python3 demo_txt2image_cosmos.py "$PROMPT" --negative-prompt="$NEGATIVE_PROMPT" --hf-token=$HF_TOKEN
```

#### 2. 根据初始视频条件和文本提示生成视频

##### 运行 Cosmos-Predict2-2B-Video2World（仅启用 PyTorch 后端）

```bash
# BF16
python3 demo_vid2world_cosmos.py "$PROMPT" --negative-prompt="$NEGATIVE_PROMPT" --hf-token=$HF_TOKEN
```


### 为 ONNX 模型和 TensorRT 引擎指定自定义路径（仅限 FLUX、Stable Diffusion 3.5 和 Cosmos）

可以使用以下方式提供预导出 ONNX 模型文件的自定义覆盖路径 `--custom-onnx-paths`这些 ONNX 模型直接用于构建 TRT 引擎，无需对 ONNX 图进行进一步优化。路径应为以逗号分隔的 <model_name> 列表：<path> 成对出现。例如： `--custom-onnx-paths=transformer:/path/to/transformer.onnx,vae:/path/to/vae.onnx`。 称呼 <PipelineClass>使用 .get_model_names(...) 获取支持的模型名称列表。

可以使用以下方式提供预构建引擎文件的自定义覆盖路径 `--custom-engine-paths`路径应为以逗号分隔的 <model_name> 列表：<path> 成对出现。例如： `--custom-onnx-paths=transformer:/path/to/transformer.plan,vae:/path/to/vae.plan`。

### 使用 Wan 从文本提示生成视频

运行以下命令，使用 Wan 2.2 生成 81 帧 720×1280 分辨率的视频。由于此型号对内存要求较高，建议启用 `--low-vram` 使用 Blackwell 设备。

```bash
# Default (81 frames, 720x1280) with --low-vram enabled
python3 demo_txt2vid_wan.py "A serene bamboo forest with sunlight filtering through the leaves" --hf-token=$HF_TOKEN --low-vram

# Adjust denoising steps, guidance scales, negative prompt, seed, warmup runs
python3 demo_txt2vid_wan.py "Ocean waves crashing on a beach at sunset" --hf-token=$HF_TOKEN --low-vram --denoising-steps 50 --guidance-scale 4.5 --guidance-scale-2 3.5 --negative-prompt "blurry, low quality, static" --seed 42 --num-warmup-runs 0
```

## 配置选项

- 可以使用以下方式设置噪声调度器 `--scheduler <scheduler>`注意：并非所有调度程序都适用于每个版本。
- 为了加快发动机制造时间 `--timing-cache <path to cache file>`如果缓存文件尚不存在，则会创建该文件。请注意，如果在多个 GPU 目标上使用缓存文件，性能可能会下降。建议仅在开发期间使用计时缓存。为了在部署时获得最佳性能，请构建不带计时缓存的引擎。
- 在切换版本、LoRa、ControlNet 等时，指定用于存储 ONNX 和引擎文件的新目录。这可以通过以下方式完成： `--onnx-dir <new onnx dir>` 和 `--engine-dir <new engine dir>`。
- 启用此功能可以提高推理性能。 [CUDA graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) 使用 `--use-cuda-graph`启用 CUDA 图需要固定的输入形状，因此该标志必须与以下参数结合使用： `--build-static-batch` 且不能与 `--build-dynamic-shape`。


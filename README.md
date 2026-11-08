# InfiniCore / InfiniLM：MiniMax-H3 海光 DCU 测评指南

本仓库用于在 **InfiniFabric 单节点 8 卡海光 BW（gfx936）容器**中构建 InfiniCore、InfiniLM，并运行 **MiniMax-H3 FL2VA（首帧图像 + 文本生成视频/音频）** 测评。

本文提供从克隆代码、安装依赖、编译框架到执行 **Legacy（PyTorch 参考实现）**、**Native（InfiniLM 实现）**及视频相似度对比的命令。以下假设网络下载和软件包安装正常，不包含额外的安装验证或故障排查步骤。

> **前提**：测评容器已经配置可用的海光驱动、DTK/HIP、海光适配版 **PyTorch 2.9.0** 和 8 张 DCU。**不要通过通用 PyPI 的 `torch` 包覆盖容器提供的海光适配版 PyTorch。** MiniMax-H3 模型权重不在本仓库中，需使用平台共享路径或自行准备。

## 1. 目录与环境要求

仓库目录：

```text
Projects/
├── InfiniCore/          # 计算、运行时、通信及 Python 接口
├── InfiniLM/            # MiniMax-H3 推理实现与测评脚本
└── Golde33443.jpg       # 本仓库提供的 FL2VA 首帧
```

建议使用以下已使用过的环境配置：

| 组件 | 版本 / 要求 |
|---|---|
| 操作系统 | Linux x86_64 |
| GPU | 8 × 海光 BW，gfx936 |
| Python | 3.11.9 |
| PyTorch | 2.9.0，海光适配构建（由平台容器提供） |
| HIP | 6.3.26093（与平台镜像匹配） |
| DTK | 25.10.0-0，路径 `/opt/dtk` |
| 编译器 | Clang 17 或兼容的 C++17 编译器 |
| xmake | 3.0.5 |
| 构建依赖 | Boost（原恢复环境为 1.92.0）、pybind11、OpenMP、CMake |
| 模型 | MiniMax-H3 的 Diffusers 格式权重，含 `diffusers/model_index.json` |

下文中的命令均在同一个 Bash 会话中依次执行。

## 2. 获取仓库并配置路径

在计划存放项目的工作目录执行：

```bash
export WS="$(pwd -P)"
git clone https://github.com/superleevv/Projects.git "$WS/Projects"

export CORE="$WS/Projects/InfiniCore"
export LM="$WS/Projects/InfiniLM"
export INFINI_ROOT="$WS/.infini"

export DTK_ROOT=/opt/dtk
export HIP_PATH=/opt/dtk/hip
export ROCM_PATH=/opt/dtk
export HYGON_CUDA_HOME=/opt/dtk/cuda/cuda
export CUDA_HOME="$HYGON_CUDA_HOME"
export CUDA_PATH="$CUDA_HOME"
export HYGON_ARCH=gfx936
export PATH="$CUDA_HOME/bin:$PATH"

export PYTHON=python3
export XMAKE_ROOT=y
export INFINILM_CXX11_ABI="$(python3 -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')"

# 在 InfiniFabric 上可使用下方共享模型路径；
# 如测评容器的挂载路径不同，仅修改 H3_MODEL。
export H3_MODEL="/public/home/aclsylqidf/shared/MiniMax-H3"
```

`H3_MODEL` 应指向 **MiniMax-H3 根目录**，实际传给推理脚本的是 `"$H3_MODEL/diffusers"`。首次拉取的是当前仓库 `main` 分支；当前仓库已包含所需的 `third_party/spdlog`、`third_party/json` 代码，不需要把仓库内的 `InfiniCore`、`InfiniLM` 当作独立 Git 仓库再次克隆。

## 3. 安装系统构建工具和 xmake

如果是 Debian/Ubuntu 系容器，且具有软件包安装权限，可安装通用构建工具：

```bash
apt-get update
apt-get install -y \
  git curl wget ca-certificates \
  build-essential clang cmake ninja-build pkg-config \
  python3-dev libomp-dev \
  libjpeg-dev libpng-dev zlib1g-dev \
  ffmpeg
```

安装指定版本的 xmake：

```bash
curl -fsSL https://xmake.io/shget.text | bash -s v3.0.5
source "$HOME/.xmake/profile"
```

**Boost、pybind11、OpenMP** 由 `InfiniCore/xmake.lua` 管理。正常联网构建时，xmake 会解析和安装其所需的构建包；Boost 的历史恢复版本为 **1.92.0**，但当前 `xmake.lua` 的 `add_requires("boost")` 未将版本写死。`spdlog` 与 `nlohmann/json` 已随仓库提供。

## 4. 安装全部 Python 依赖

Python 环境沿用容器预装的海光 PyTorch。以下版本优先按照 `restore_minimax_h3_unified_fast.sh` 的恢复记录固定，并补充 FL2VA 视频评测需要的依赖。

### 4.1 FL2VA 主依赖及兼容包

```bash
python3 -m pip install --user \
  janus==2.0.0 \
  xxhash==3.6.0 \
  diffusers==0.40.0 \
  accelerate==1.14.0 \
  av==16.0.1 \
  httpx==0.28.1 \
  httpcore==1.0.9 \
  h11==0.16.0 \
  anyio==4.15.1 \
  certifi==2026.10.23 \
  idna==3.20 \
  typing_extensions==4.16.0 \
  safetensors==0.8.0 \
  transformers==5.14.1 \
  huggingface-hub==1.28.0 \
  click==8.4.2 \
  hf-xet==1.5.2 \
  typer==0.20.0 \
  rich==14.2.0 \
  shellingham==1.5.4 \
  markdown-it-py==4.2.0 \
  Pygments==2.21.0 \
  mdurl==0.1.2 \
  "tokenizers>=0.22.0,<=0.23.0" \
  regex==2026.5.9 \
  einops==0.8.2
```

这里使用正常的 pip 依赖解析，为上述依赖自动安装必要的间接包。`transformers==5.14.1` 对 Tokenizers 有版本范围约束，因此没有沿用早期环境中的 `tokenizers==0.20.1`。

### 4.2 视频编码及对比评分依赖

```bash
python3 -m pip install --user \
  numpy==1.25.0 \
  Pillow==12.2.0 \
  scipy==1.11.4 \
  scikit-image==0.25.2 \
  lazy-loader==0.6 \
  imageio==2.37.0 \
  tifffile==2024.9.20 \
  networkx==3.6.1 \
  packaging==26.0 \
  imageio-ffmpeg \
  pytest==9.1.1
```

视频生成使用 `av`（PyAV）进行 MP4 编码；视频比较脚本使用 `av`、NumPy、Pillow、`scikit-image` 计算帧级 SSIM。其余为相关兼容、视频处理及测试工具依赖。

### 4.3 安装海光专用 Triton 和 FlashAttention

**必须使用海光软件包源**，不要把这里的包替换成普通 CUDA/NVIDIA wheel。

```bash
python3 -m pip install --user --no-deps \
  --no-cache-dir --only-binary=:all: \
  --index-url https://pypi.sourcefind.cn/release/dtk/+simple/ \
  "triton==3.5.1+das.opt1.dtk2604.torch290" \
  "flash-attn==2.6.1+das.opt1.dtk2604.torch290"
```

### 4.4 安装适配海光 PyTorch 的 torchvision 0.24.0

`torchvision` 涉及编译型算子，应使用与平台 PyTorch/DTK 匹配的版本。若容器**未预装兼容构建**，可从源码构建：

```bash
git clone --depth 1 --branch v0.24.0 \
  https://github.com/pytorch/vision.git \
  "$WS/vision-v0.24.0"

BUILD_VERSION=0.24.0 \
FORCE_CUDA=1 \
CUDA_HOME="$CUDA_HOME" \
MAX_JOBS=8 \
python3 -m pip install --user \
  --no-deps --no-build-isolation \
  "$WS/vision-v0.24.0"
```

此处的 `FORCE_CUDA=1` 采用海光 DTK 的 CUDA 兼容工具链编译，不表示应另装 NVIDIA CUDA。已提供适配版 `torchvision==0.24.0` 的容器可跳过该源码安装步骤。

## 5. 编译安装 InfiniCore

```bash
export TORCH_LIB="$(python3 -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$CORE"

xmake f -y -m release \
  --hygon-dcu=y \
  --hygon-arch=gfx936 \
  --cuda="$CUDA_HOME" \
  --ccl=y \
  --aten=y \
  --omp=y \
  --ccache=n

# 安装 InfiniRT / InfiniOP / InfiniCCL 及 C++ 接口
xmake build -j 16 infinicore_cpp_api
xmake install -y

# 构建并安装 Python 扩展
xmake build -j 16 _infinicore
xmake install -y _infinicore

python3 -m pip install --user -e "$CORE" --no-deps --no-build-isolation
```

编译选项中的 `--ccl=y` 要求容器的海光通信库/NCCL 兼容接口可用。

## 6. 编译安装 InfiniLM

```bash
cd "$LM"

xmake f -y -m release
xmake build -j 16 _infinilm
xmake install -y _infinilm

python3 -m pip install --user -e "$LM" --no-deps --no-build-isolation

export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$CORE/build/linux/x86_64/release:$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$LM/python:$CORE/python${PYTHONPATH:+:$PYTHONPATH}"
```

`INFINI_ROOT` 是两套框架共用的头文件和动态库安装位置；`INFINILM_CXX11_ABI` 使用第 2 节中从海光 PyTorch 取得的 ABI 设置。

## 7. 运行 MiniMax-H3 FL2VA

以下使用仓库自带的 `Golde33443.jpg`，在 **同一首帧、同一提示词、同一种子及生成参数**下分别运行 Legacy 和 Native。两次执行应分别独立启动，以便测评人员查看输出和终端计时信息。

```bash
export FIRST_FRAME="$WS/Projects/Golde33443.jpg"
export PROMPT="A dog falls asleep"
cd "$LM"
```

### 7.1 Legacy：纯 PyTorch 参考实现

```bash
python3 examples/minimax_h3.py fl2va \
  --model-path="$H3_MODEL/diffusers" \
  --text="$PROMPT" \
  --first-frame="$FIRST_FRAME" \
  --gpus=0,1,2,3,4,5,6,7 \
  --legacy \
  --legacy-text-gpus=0,1,2,3 \
  --legacy-transformer-gpus=4,5,6,7 \
  --num-frames=124 \
  --steps=10 \
  --seed=42 \
  --height=384 \
  --width=576 \
  --output="$WS/minimax_h3_fl2va_legacy.mp4"
```

### 7.2 Native：InfiniLM 实现

```bash
python3 examples/minimax_h3.py fl2va \
  --model-path="$H3_MODEL/diffusers" \
  --text="$PROMPT" \
  --first-frame="$FIRST_FRAME" \
  --gpus=0,1,2,3,4,5,6,7 \
  --num-frames=124 \
  --steps=10 \
  --seed=42 \
  --height=384 \
  --width=576 \
  --output="$WS/minimax_h3_fl2va_native.mp4"
```

上述两条命令仅以 `--legacy` 和 Legacy 专属设备分配参数区分运行后端。若正式测评要求使用 **50 steps**，则将两条命令中的 `--steps=10` **同时改为 `--steps=50`**，保持其余参数一致。

## 8. 对比两份生成视频

调用仓库自带的 `examples/minimax_h3_compare.py`，直接对已经生成的两份 MP4 评分，不再重新推理：

```bash
cd "$LM"

python3 examples/minimax_h3_compare.py \
  --model-path="$H3_MODEL/diffusers" \
  --tasks=fl2va \
  --reference-fl2va-video="$WS/minimax_h3_fl2va_legacy.mp4" \
  --infinilm-fl2va-video="$WS/minimax_h3_fl2va_native.mp4" \
  --output-dir="$WS/minimax_h3_comparison"
```

脚本输出包括视频帧尺寸、解码帧数及平均 SSIM。**完成视频生成不等于通过输出质量测评**；应以生成文件及比较结果为准。

## 9. 依赖版本说明

本 README 中固定版本主要取自已使用的 `restore_minimax_h3_unified_fast.sh`：

- **13 项基础 Python 包**：`janus`、`xxhash`、`diffusers`、`accelerate`、`av`、`httpx`、`httpcore`、`h11`、`anyio`、`certifi`、`idna`、`typing_extensions`、`safetensors`。
- **11 项兼容包**：`transformers`、`huggingface-hub`、`torchvision`、`click`、`hf-xet`、`typer`、`rich`、`shellingham`、`markdown-it-py`、`Pygments`、`mdurl`。
- **2 项海光专用包**：`triton`、`flash-attn`。
- **其他显式安装项**：`tokenizers`、`regex`、`einops`，以及第 4.2 节中的视频处理、评分和测试依赖。
- **平台基础依赖**：海光适配版 PyTorch、HIP、DTK、驱动和通信运行库由测评容器提供，不能照搬普通 CUDA 发行包。
- **构建依赖**：xmake 3.0.5；Boost、pybind11、OpenMP 通过 xmake 配置及系统编译工具链提供。

由于原恢复脚本在旧容器中直接使用了 AFS 内保存的 Wheel 和构建产物，本 README 将其转换为**面向全新容器的公开下载/源码编译流程**，不依赖原个人 AFS 存储路径。**该新容器安装流程尚未作为一条完整流水线重新实测。**

## 10. 项目参考

- [InfiniCore 源码与构建说明](./InfiniCore/README.md)
- [InfiniLM 源码与构建说明](./InfiniLM/README.md)
- [FL2VA 推理入口](./InfiniLM/examples/minimax_h3.py)
- [Legacy / Native 视频对比入口](./InfiniLM/examples/minimax_h3_compare.py)
- [Diffusers MiniMax-H3 支持](https://github.com/huggingface/diffusers/releases/tag/v0.40.0)
- [xmake 安装指南](https://xmake.io/guide/quick-start.html)

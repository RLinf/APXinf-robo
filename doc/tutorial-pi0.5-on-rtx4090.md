# 从零部署 π0.5：基于 APXInf 跑通 VLA 模型推理全流程
——by @[卜寒兮 - 知乎](https://www.zhihu.com/people/spark-20-22)


这两年 VLA 模型的进展很快，比如 π0.5、GR00T 等模型不断更新。但是对于想实际尝试的开发者来说，别人的模型开源之后你下载下来，这只是开始。

部署环境要怎么配置，加载权重和准备配套文件，怎样用 Python 真正调用一次推理实际跑通，再判断返回结果是否正常？这些实际的问题，需要沿着部署流程走一遍才能搞清楚。

这篇文章记录了我以 APXInf 为入口，在 RTX 4090 上运行 π0.5 的全过程，包括环境准备、源码编译、真实权重首次推理等，也包括部署过程中踩到的坑。其中用到的一些脚本，我也会分享出来，希望能给同样有 VLA 部署需求的知友提供一点参考。

项目地址：[RLinf/APXinf-robo](https://github.com/RLinf/APXinf-robo)

通过开源推理引擎 APXInf 加载 π0.5，并用 Python 完成了从输入图像和指令，到返回动作数据的首次调用。在本文的测试配置下（具体参考后文），预热后的完整推理调用耗时约为 33 ms，每次输出 10 步动作。

![](https://obs-notes-pics.oss-cn-shanghai.aliyuncs.com/PicGo/PixPin_2026-10-26_12-10-22.png)

## 先明确要跑通什么

首先简单介绍一下 APXInf 这个项目，它是一个面向 VLA 模型部署与推理的优化引擎，提供已有模型的推理实现和 Python 接口。本文使用的仓库是 `APXinf-robo`（https://github.com/RLinf/APXinf-robo），它将底层引擎与机器人相关的输入输出约定、接入和评测工具连接起来。对于会写 Python、想尝试 VLA 部署的开发者，可以从它现成的模型和调用入口开始。

VLA 是 Vision-Language-Action，也就是视觉、语言与动作。可以把 VLA 模型的工作理解为先接收图像、任务指令，以及配置要求的状态信息，然后生成一段机器人动作。

![](https://obs-notes-pics.oss-cn-shanghai.aliyuncs.com/PicGo/20261024151854334.png)

在开发项目里，APXInf 承担的是这一段**从观察输入到动作输出的推理工作**。已有 Python 程序可以组织观察字典，调用 Policy 完成预处理、模型计算和动作还原，再把结果交给原有执行流程。如果项目使用 OpenPI 客户端，仓库还提供 OpenPI 兼容的 WebSocket 服务，可作为进一步接入的入口。

![](https://obs-notes-pics.oss-cn-shanghai.aliyuncs.com/PicGo/4871c4af-d7a7-47ea-9c06-888163f06146.png)

这篇文章的目标是安装 APXInf，加载真实 π0.5 权重，用 Python 完成一次推理，并保存和检查结果。输入采用两张黑色图像、零状态和一条任务指令，用来检查调用链路。机器人仿真、真实场景输入与任务成功率评测，留到后续。

本文会用到的脚本我放到了`https://github.com/zxw1992/APXinf-scripts-on4090`。

其中包含了 CUDA 组件安装工具、报错时需要的补丁、最小推理示例和测量程序等。可以下载下来保存到`~/apxinf-runtime/bootstrap`目录下（后文会提）以供复现过程中使用。
## 环境准备，完成源码安装

本文是按照从零搭建环境的情况给出的详细上手步骤，如果有的软件已经安装过，跳过对应的安装命令即可。
### 一、安装依赖和创建工作目录
APXinf 比较友好的一点是可以在 4090 这种消费级显卡上部署。我这次用的就是一张 4090 单卡的机子，具体软硬件配置如下：

| 项目              | 版本                                         |
| --------------- | ------------------------------------------ |
| 系统              | Ubuntu 22.04.4 LTS，x86_64                  |
| GPU             | RTX 4090，24GB 显存                           |
| 驱动              | 580.95.05                                  |
| CUDA 开发组件       | 12.8 Update 1，nvcc 12.8.93，cuBLAS 12.8.4.1 |
| Python          | 3.11.8，独立 venv                             |
| Rust / Cargo    | 1.95.0                                     |
| CMake / maturin | 3.31.10 / 1.15.0                           |

1、如果机子上还没有下面这些基础的编译、git及下载解压所需的基础依赖工具， 可以先安装一下，打开终端，执行：
```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential binutils pkg-config \
  git curl ca-certificates tar xz-utils
```

2、创建并进入工作目录：

```bash
cd ~
mkdir -p apxinf-runtime
cd apxinf-runtime
```

这里为了避免后文反复定义环境变量和创建路径，可以把下面这段内容保存为 `~/apxinf-runtime/env.sh`：

```bash
#!/usr/bin/env bash
# 使用 source 加载本文件。

# 工作目录与模型路径
export APX_RUNTIME="${APX_RUNTIME:-$HOME/apxinf-runtime}"
export APX_PROJECT="$APX_RUNTIME/APXinf-robo"
export APX_VENV="$APX_RUNTIME/venv"
export APX_MODEL_DIR="$APX_RUNTIME/models/pi05_libero_base-5fcf06f"

# 编译配置
export CARGO_TARGET_DIR="$APX_RUNTIME/target/wheel"
export RUSTUP_TOOLCHAIN=1.95.0

# CUDA：使用本文配套脚本的安装目录
export CUDA_HOME="$APX_RUNTIME/cuda-12.8"
export CUDA_PATH="$CUDA_HOME"

# 首次执行时，虚拟环境可能尚未创建
if [ -f "$APX_VENV/bin/activate" ]; then
    . "$APX_VENV/bin/activate"
else
    printf '%s\n' 'venv 尚未创建；创建后请重新 source 本文件。'
fi

# 在激活虚拟环境之后补充工具路径
export PATH="$CUDA_HOME/bin:$HOME/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 创建必要目录
mkdir -p \
  "$APX_RUNTIME/bootstrap" \
  "$APX_RUNTIME/models" \
  "$APX_RUNTIME/results" \
  "$CARGO_TARGET_DIR"
```

然后加载：

```bash
source "$HOME/apxinf-runtime/env.sh"
```

这里面两个重要的目录，
- 一个是`APX_RUNTIME` ：工作目录
- 一个是`APX_PROJECT` ：稍后存放源码的目录
- `$APX_RUNTIME/bootstrap`、 `$APX_RUNTIME/models` 、`$APX_RUNTIME/results`是方便后面保存各种文件而创建的目录

比如，刚才提到的脚本下载下来保存到`$APX_RUNTIME/bootstrap`目录下。

3、然后可以`nvidia-smi`检查一下显卡和驱动版本:

![](https://obs-notes-pics.oss-cn-shanghai.aliyuncs.com/PicGo/PixPin_2026-10-24_16-23-00.png)

如果看不到 GPU 信息，需要先检查是不是驱动有问题。

### 二、创建环境，编译并安装

1、Python 环境和构建工具：

本次实测使用的 python 版本是 3.11.8。如果你的机子还没有 python 环境，可以先完成以下安装。

```bash
# 安装 Python 3.11 及配套组件
sudo apt-get update
sudo apt-get install -y software-properties-common

sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update

sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
```

安装后检查：

```
python3.11 --version
```

应输出 `Python 3.11.x`

本次部署，建议新建一个 Python 虚拟环境，在独立环境中进行：

```bash
cd "$APX_RUNTIME"
python3.11 -m venv venv
source "$APX_RUNTIME/env.sh"
```

然后在已经激活的环境中安装构建工具：

```bash
python -m pip install --upgrade pip setuptools wheel 'maturin==1.15.0' 'cmake==3.31.10'
```

安装 Rust 工具链，从 Rust 官方获取安装器并安装:

```bash
cd "$APX_RUNTIME/bootstrap"
curl --proto '=https' --tlsv1.2 -fL --retry 3 \
  https://sh.rustup.rs -o rustup-init.sh
sh rustup-init.sh -y --no-modify-path --profile minimal --default-toolchain 1.95.0
```


2、接下来比较重要的是 cuda 开发环境，`nvidia-smi` 显示驱动支持的 CUDA 版本，并不表示已装好 `nvcc`。

由于这个项目需要在本地编译，所以要保证机器上安装好 nvcc，按照官方在 GitHub 上的说明，cuda 套件的建议版本是12.8。

如果已经安装，可直接到后面的版本验证；如果没有安装，可以使用下面这个脚本进行安装：

```python
python3 "$APX_RUNTIME/bootstrap/install_cuda_components.py"
```

![](https://obs-notes-pics.oss-cn-shanghai.aliyuncs.com/PicGo/PixPin_2026-10-24_17-18-53.png)

由于我之前已经安装过，所以截图里没有显示进程。

使用本文提供的脚本安装时，CUDA 开发组件位于 `$APX_RUNTIME/cuda-12.8`；如果你机子上已有，就把上面 `env.sh` 中的 `CUDA_HOME` 改为实际安装目录，例如 `/usr/local/cuda-12.8`，然后重新执行 `source "$APX_RUNTIME/env.sh"`。

最后验证 nvcc 版本：
```bash
nvcc --version
```

如果出现如下结果，说明安装成功：

![](https://obs-notes-pics.oss-cn-shanghai.aliyuncs.com/PicGo/PixPin_2026-10-24_17-20-24.png)

3、进入工作目录，克隆项目到源码目录：
```bash
cd "$APX_RUNTIME" 
git clone --recursive https://github.com/RLinf/APXinf-robo.git "$APX_PROJECT" 
cd "$APX_PROJECT"

# 保证源码与本文使用的一致 
git checkout --detach 835f68af3d59d48d6bf4a724191127d4434d8cbb 
git submodule update --init --recursive
```

4、编译项目，安装原生扩展、引擎 Python 前端和 Robo 包：
```bash
maturin build --release --locked --features cuda --auditwheel skip -m apxinf/crates/apxinf-py/Cargo.toml
python -m pip install --force-reinstall "$CARGO_TARGET_DIR"/wheels/apxinf_py-*.whl
python -m pip install -e ./apxinf/python/apxinf --config-settings editable_mode=strict
python -m pip install -e .
```

三次 pip 安装分别对应原生扩展、引擎 Python 前端和 Robo；NumPy 等普通 Python 依赖会按包声明安装。成功后应能看到生成的 wheel 和 pip 安装完成信息。类似如下截图中的提示，首次编译的时候会慢一些，耐心等待即可：
![](https://obs-notes-pics.oss-cn-shanghai.aliyuncs.com/PicGo/PixPin_2026-10-24_21-00-08.png)

5、 确认当前 Python 环境能否加载项目的三个包：

```bash
python -c 'import apxinf_py, apxinf, apxinf_robo; print("IMPORT_OK", apxinf_py.__version__, apxinf_robo.__version__)'
```

如果看到`IMPORT_OK 0.1.0 0.1.0`提示信息，就说明导入没问题。

![](https://obs-notes-pics.oss-cn-shanghai.aliyuncs.com/PicGo/PixPin_2026-10-24_21-29-38.png)

但是，我在第一次编译后导入的时候遇到了如下报错：

```
undefined symbol: apxinf_flashinfer_gdn_launch
```

![](https://obs-notes-pics.oss-cn-shanghai.aliyuncs.com/PicGo/PixPin_2026-10-26_16-54-52.png)

我用 AI 帮我查了一下，是因为本次实测使用的APXinf 引擎（commit 提交为https://github.com/infinigence/ApxInf/tree/7f37642dd6c5896bca66d7640d9944c2a3fe04b4）跟 4090 兼容性的问题，SM89 没有编译面向 Blackwell 的 GDN launcher，但普通适配层仍然引用该符号，导致 wheel 无法导入。π0.5 不走这个 GDN 计算路径。

解决方案是打一个局部兼容补丁，补齐相同 C ABI 的入口。

如果你也遇到这个报错，可参考下面的处理方案：

先将 `sm89-gdn-compat.patch` 和 `apply_sm89_compat.sh` 两个文件直接放入 `$APX_RUNTIME/bootstrap`，然后执行：

```bash
bash "$APX_RUNTIME/bootstrap/apply_sm89_compat.sh"
```

脚本会检查 GPU 和引擎版本，并应用补丁。检查通过、处理完成后，会输出：`SM89_COMPAT_READY`。

如果版本检查或补丁检查失败，先核对源码版本，暂时不要继续下面的编译。

接着重新加载环境，重新编译并安装原生扩展：

```bash
source "$APX_RUNTIME/env.sh"
cd "$APX_PROJECT"

maturin build --release --locked --features cuda --auditwheel skip \
  -m apxinf/crates/apxinf-py/Cargo.toml

python -m pip install --force-reinstall \
  "$CARGO_TARGET_DIR"/wheels/apxinf_py-*.whl
```

这里重新加载 `env.sh`，是为了让`apply_sm89_compat.sh`脚本写入的 `APXINF_SM89_GDN_COMPAT=1` 在当前终端生效。

完成后再次检查：

```bash
python -c 'import apxinf_py, apxinf, apxinf_robo; print("IMPORT_OK", apxinf_py.__version__, apxinf_robo.__version__)'
```

看到 `IMPORT_OK` 后，说明报错问题解决了，继续下面的步骤。

6、然后运行官方仓库自带的随机权重检查程序：

```bash
cd "$APX_PROJECT"
python scripts/bench_pi05.py --random-weights --precision bf16 --layer l1 \
  --views 2 --image-size 224 --token-count 10 \
  --action-horizon 10 --num-flow-steps 10 \
  --warmup 2 --samples 5
```

`--layer l1` 意思是测量底层 ModelRunner 的调用耗时，包含 Python/Rust 调用以及从已调整尺寸的 RGB 图像开始的推理。

这里在官方给出的示例基础上，配置两路 224×224 图像，使用 bf16，动作块长度和内部生成步数都设为 10。程序先预热两次，再记录五次调用耗时。

![](https://obs-notes-pics.oss-cn-shanghai.aliyuncs.com/PicGo/PixPin_2026-10-24_21-45-10.png)

输出显示已经正确识别到 4090，L1 调用的中位耗时为 31.27 毫秒，说明这组配置的执行链路已经跑通。

如果你顺利进行到这里，说明环境配置、源码的构建和 GPU 执行链路都走通了。可以加载真实模型权重进行推理了。

## 基于 π0.5，完成 APXinf  下的首次推理
1、下载模型权重和配套文件

本次使用的 checkpoint 是 `lerobot/pi05_libero_base`。

终端下载可以使用如下命令：

```bash
# 创建并进入模型目录
mkdir -p "$APX_MODEL_DIR"
cd "$APX_MODEL_DIR"

# 固定模型版本的下载地址
APX_MODEL_URL='https://hf-mirror.com/lerobot/pi05_libero_base/resolve/5fcf06fc42b6717e22ab885fd381a096574b12c7'

# 下载配置和权重
curl -fL "$APX_MODEL_URL/config.json" -o config.json
curl -fL -C - "$APX_MODEL_URL/model.safetensors" -o model.safetensors

# 下载归一化统计和 tokenizer
curl -fL https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json -o norm_stats.json
curl -fL https://storage.googleapis.com/big_vision/paligemma_tokenizer.model -o tokenizer.model
```

这里`tokenizer.model`和`norm_stats.json` 走的是Google Storage 地址，如果因网络环境问题导致下载不成功，可以使用如下备用地址：

```bash
curl -fL 'https://hf-mirror.com/Fisher-Wang/pi05-libero-pytorch/resolve/ab107fbe6661e8abfec0f13dc27bfd3fafe24c69/assets/physical-intelligence/libero/norm_stats.json' -o norm_stats.json
curl -fL 'https://hf-mirror.com/RLinf/openpi_tokenizer/resolve/befaa248e4f82954b625a421658f933dfd1a97a0/big_vision/paligemma_tokenizer.model' -o tokenizer.model
```

完成后你将获得 4 个文件：

| 文件                  | 用途                          |
| ------------------- | --------------------------- |
| `model.safetensors` | 模型权重，约 14.4 GB              |
| `config.json`       | 模型配置                        |
| `tokenizer.model`   | 处理文本指令                      |
| `norm_stats.json`   | 本次 LIBERO 配置对应的归一化统计，用于动作还原 |

具体耗时就取决于你的网速了。

![](https://obs-notes-pics.oss-cn-shanghai.aliyuncs.com/PicGo/PixPin_2026-10-24_22-15-15.png)

2、用一个完整的小程序调用模型

下面是基于项目的 `build_robot_policy` 和 `policy.infer` 写的一个模型调用脚本`quickstart_real.py`，增加了结果检查与文件保存，便于跑完以后留下可核验的产物。

```python
#!/usr/bin/env python3
"""A small real-weight API example. Black frames test the interface, not a task."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from apxinf_robo import build_robot_policy

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-dir", required=True, type=Path)
parser.add_argument("--out-dir", required=True, type=Path)
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=False)

policy = build_robot_policy(
    "franka_libero", args.model_dir, precision="bf16", device="cuda:0",
    action_horizon=10, num_flow_steps=10, discrete_state=False, seed=0,
    norm_stats=args.model_dir / "norm_stats.json",
    tokenizer_path=args.model_dir / "tokenizer.model",
)
try:
    metadata = policy.metadata
    observation = {key: np.zeros((256, 256, 3), np.uint8) for key in metadata["image_keys"]}
    observation[metadata.get("prompt_key", "prompt")] = "put both moka pots on the stove"
    if metadata.get("state_key"):
        observation[metadata["state_key"]] = np.zeros(metadata["state_dim"], np.float32)
    started = time.perf_counter()
    result = policy.infer(observation)
    wall_ms = (time.perf_counter() - started) * 1000
    actions = np.asarray(result["actions"])
    assert actions.shape == (10, 7)
    assert actions.dtype == np.float32 and np.isfinite(actions).all()
    np.save(args.out_dir / "actions.npy", actions, allow_pickle=False)
    summary = {"scope": "real weights, synthetic input, API check only",
               "shape": list(actions.shape), "dtype": str(actions.dtype), "finite": True,
               "first_action": actions[0].tolist(), "first_call_wall_ms": wall_ms,
               "timing": result.get("timing", {}), "metadata": metadata}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print("QUICKSTART_REAL_WEIGHTS_OK", flush=True)
finally:
    policy.close()
```

这里面需要说明的几个参数是：
- `franka_libero` 选择输入输出预设；
- `action_horizon=10` 表示输出 10 个时间步；
- `num_flow_steps=10` 是生成动作时使用的内部迭代步数
- 图像字段从 `metadata` 读取，本次有两路：输入是 256×256 RGB 图，预处理在调用链路内完成。
- 示例使用黑色图像、8 维零状态和固定提示词作为输入。

如果已经下载了上文提到的配套脚本，那直接运行即可，或者可以手动保存在`bootstrap`目录下。运行命令如下（btw，建议每次使用一个新输出目录，避免覆盖上次结果）

```bash
cd "$APX_RUNTIME"
export APX_RESULT_DIR="$APX_RUNTIME/results/quickstart-$(date -u +%Y%m%d-%H%M%S)"
python -u "$APX_RUNTIME/bootstrap/quickstart_real.py" \
  --model-dir "$APX_MODEL_DIR" --out-dir "$APX_RESULT_DIR"
```

成功时会打印动作形状、类型、有限值检查和第一行动作，最后出现`QUICKSTART_REAL_WEIGHTS_OK`。

看到这个标志，并列出 `actions.npy` 与 `summary.json`两个文件，意味着成功完成了真实模型的首次推理。上述两个文件前者保存动作，后者记录参数与计时。

![](https://obs-notes-pics.oss-cn-shanghai.aliyuncs.com/PicGo/PixPin_2026-10-24_23-04-24.png)
![](https://obs-notes-pics.oss-cn-shanghai.aliyuncs.com/PicGo/PixPin_2026-10-24_23-14-54.png)

输出的 `shape: [10, 7]`表示 10 个时间步，每步 7 维动作，不过这里`first_action` 只展示第一步，完整的 10 步动作保存在 `actions.npy` 中。

本次首次调用耗时约 146.58 ms，这是不含模型加载的时间，模型加载差不多一分多钟。

我在复现的时候还做了一个推理耗时测试，就是用同一组测试输入反复调用模型（10 次预热和 30 次测量），看看稳定运行后，一次推理通常要多久。

![](https://obs-notes-pics.oss-cn-shanghai.aliyuncs.com/PicGo/PixPin_2026-10-24_23-39-51.png)

测试结果显示，模型加载与初始化耗时约 96.6 秒，首次调用约 146.88 ms；预热10 次后，完整推理调用的时间稳定在 33 ms 左右。

也就是说，如果模型加载后持续接收请求，这个 33 ms 更能反映它在这组测试输入下的日常运行速度。

就本次配置看，4090 可以完成真实 π0.5 权重加载，预热后完整调用的推理耗时约 33 毫秒，每次生成 10 步动作。

## 跟其他框架的对比
为了比较一下 APXinf 的加速效果，我还试了一下在同一台 4090 上，用同一份 π0.5 权重，跑了一组它在 PyTorch 框架下的推理耗时。

具体来说是同一份真实 π0.5 权重下完成三组测试，各 3 轮、每轮首次调用后预热 10 次，再正式测量 100 次。

![](https://obs-notes-pics.oss-cn-shanghai.aliyuncs.com/PicGo/PixPin_2026-10-25_01-41-06.png)

测试结果显示在 pytorch 框架下，同样的模型PyTorch 普通运行一次推理的用时在 171 ～ 179 ms 区间内。编译优化后的 PyTorch＋torch.compile 集中在 49ms 左右。

![](https://obs-notes-pics.oss-cn-shanghai.aliyuncs.com/PicGo/PixPin_2026-10-25_01-48-50.png)

APXInf 相对普通 PyTorch 的 P50 加速比为 **5.28×**，相对 torch.compile 的 P50 加速比为 **1.46×**。对应耗时分别减少 81.1% 和 31.4%。

另外，在“Real-Time Execution of Action Chunking Flow Policies”这篇论文中，同样在相同配置的显卡上，报告的普通 π0.5 为 **76 ms**，启用 RTC 后为 **97 ms**，不过这篇文章用的是 5 步生成、且仅计 GPU 计算的平均值。

![](https://obs-notes-pics.oss-cn-shanghai.aliyuncs.com/PicGo/PixPin_2026-10-25_01-50-11.png)

具体对比可参看下图：

![](https://obs-notes-pics.oss-cn-shanghai.aliyuncs.com/PicGo/PixPin_2026-10-25_01-53-44.png)

总体来看，APXInf 的加速效果还是挺明显的，尤其适合在模型长期驻留、持续生成动作块的场景中。

不过还是要强调一下的是，这个测试只能代表仿真条件下的表现，当接入机器人，真实执行任务的时候，那就是另外的情况了。因为机器人整体控制频率还受采集、通信、执行和重规划周期影响，需要在具体系统中继续测量。

## 总结
总体来说，APXinf 的部署和复现并不是特别复杂，至少我没有踩到特别大的坑，还是挺顺利的，可能最大的问题就是工具安装和模型下载会遇到网络不通畅的情况。比如 Rust 下载超时，这种换个镜像会好点。

对于实际运行阶段，Python 调用比较直接，首次部署的主要工作在编译环境、版本对应和资产准备。配好之后，构造字典、调用模型的代码不长。

以下，我把前面步骤总结成一份 Quick Start，大家如果按照这篇文章部署的时候可以这样核对：

| 顺序  | 要做的事                       | 可以继续的标志                    |
| --- | -------------------------- | -------------------------- |
| 1   | `cd ~`、创建工作目录，安装基础工具       | GPU 可见，各工具可输出版本            |
| 2   | 获取源码与子模块，准备 CUDA、Rust、venv | 工具版本符合要求                   |
| 3   | 编译安装                       | 三个包可导入                     |
| 4   | 小规模随机权重运行检查                | GPU 测试完成                   |
| 5   | 下载四项模型资产并校验                | 四个文件下载完成，位于 $APX_MODEL_DIR |
| 6   | 运行 `quickstart_real.py`    | 返回有限值 `(10, 7)` 动作，结果文件已保存 |

这次完成了安装、GPU 检查、真实权重首次推理和独立进程复跑。后续可以换实际观察输入，接入仿真验证任务效果。有相同学习需求的读者，可从[GitHub 仓库](https://github.com/RLinf/APXinf-robo)、[Quick Start](https://github.com/RLinf/APXinf-robo#build-apxinf-robo)和本文配套文件开始，按自己的环境逐项尝试。

遇到问题先查已有 Issue，再提供 GPU、驱动、CUDA、两个源码提交、模型 revision、精度、完整命令和报错。当然也欢迎在评论区讨论。


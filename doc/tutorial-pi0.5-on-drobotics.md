# 在 RDK S600 上用 π0.5 跑 LIBERO 评测

这篇教程说明如何把地瓜机器人 RDK S600 当作 π0.5 的推理端，接到本仓库的 LIBERO 评测器上：

```text
PC：LIBERO 仿真 + apxinf-robo eval-libero
        │  OpenPI WebSocket（图像、状态、指令 → 动作块）
        ▼
S600：apxinf-robo serve --engine drobotics
        └─ apx_drobotics：preprocess → C++ 推理进程 → 地瓜 SDK v2.0.0 → BPU → postprocess
```

板上运行的是地瓜官方的 `Pi05(libero)` W8 模型和 LLM SDK v2.0.0 运行时，**不使用 ApxInf**。ApxInf 不支持 BPU，所以 S600 这部分封装成了和 ApxInf 并列的 `apx-drobotics`，上层的 apxinf-robo 两边共用。整个仓库部署在板上，只是不需要 `apxinf/` 子模块。

PC 端的评测器和连接其他服务时一样，只要指定 `--precision w8`。PC 上也不需要安装 ApxInf。

## 准备条件

**S600 板子**

| 项目 | 要求 |
| --- | --- |
| 系统 | RDK S600 V5.1.0 及以上（地瓜对 SDK v2.0.0 的要求） |
| BPU 内存（ION） | carveout 至少约 5 GB，且能放下一块约 3.4 GB 的连续内存 |
| 磁盘 | 部署过程中约 8 GB，部署完成后约 4.4 GB |
| 网络 | 能访问 `d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com`，用于下载 SDK 和模型 |
| 软件 | `python3`（含 venv）、`g++`、`cmake`、`curl` |

ION 不够时，模型加载会报 `Memory alloc failed`。默认的 `cpu_first` 布局里 carveout 只有 2 GiB，放不下 3.32 GB 的语言模型。要在**宿主机**上执行 `/usr/hobot/bin/hb_switch_ion.sh balanced` 再重启，Docker 容器里改不了。确认方法：

```bash
grep -E "total size" /sys/kernel/debug/ion/heaps/carveout
```

**PC**

装好 LIBERO 和本仓库的评测依赖，见 README 的 LIBERO 评测部分。评测器通过 `openpi_client` 连接板子。这条路径用不到 ApxInf，所以不需要编译它。

## 第一步：在板上部署

在板上取得本仓库的代码（不需要初始化子模块），然后运行部署脚本：

```bash
bash apx-drobotics/scripts/deploy.sh        # 默认部署到 /opt/d-robotics-llm/2.0.0
```

脚本会依次完成下面这些步骤，已经做过的会跳过，所以可以放心重复运行：

1. 下载 SDK（约 1.2 GB），用契约里记录的 SHA-256 校验，只解压板端运行时。
2. 下载 `Pi05_libero.zip`（约 3.2 GB），校验后解压到 SDK 的模型目录，再删除压缩包。
3. 创建 Python 虚拟环境，安装 `apx-drobotics/requirements-board.txt` 中固定版本的依赖，再安装 apx-drobotics 和 apxinf-robo。
4. 编译 C++ 推理进程，产物在 `apx-drobotics/build/worker/`。
5. 用契约检查整个部署：SDK 库、三段 HBM、tokenizer、归一化统计、运行时配置，逐个比对大小和 SHA-256。

第一次部署的时间主要花在下载上。最后一步通过才算部署完成。

## 第二步：启动服务

```bash
/opt/d-robotics-llm/2.0.0/venv-apx-drobotics/bin/apxinf-robo serve --engine drobotics --robot franka_libero \
  --model-dir /opt/d-robotics-llm/2.0.0/D-Robotics_LLM_2.0.0_S600_S100_SDK/oellm_runtime_s600 \
  --port 8000
```

`--model-dir` 指向 SDK 的 `oellm_runtime_s600` 目录。需要改推理进程的路径或超时时，用 `--policy-options` 传 JSON，比如 `'{"infer_deadline": 10}'`。只有 ApxInf 才支持的参数（`--checkpoint`、`--autotune` 等）会直接报错，不会被悄悄忽略。

启动顺序如下：

1. 再检查一遍部署。不通过就直接退出（退出码 2），不会开始监听。
2. 启动推理进程并加载模型，约 14 秒。加载失败时退出码是 3。
3. 日志出现 `listening on ws://...` 后才开始监听。

`http://<板子>:8000/healthz` 返回 `OK` 说明服务已就绪。

按 Ctrl+C 或发 SIGTERM 会正常退出并释放 BPU 内存。推理进程如果崩溃、超时（默认 5 秒）或报错，服务会把错误告诉客户端，然后以退出码 1 退出。这时重启服务即可。

## 第三步：在 PC 上跑评测

```bash
apxinf-robo eval-libero --backend websocket --host <板子地址> --port 8000 \
  --precision w8 \
  --suite libero_object --trials-per-task 10 \
  --results-jsonl results.jsonl --summary-json summary.json
```

板子不能直接访问时，可以用 SSH 端口转发，例如 `ssh -L 8000:localhost:8000 ...`，然后把 `--host` 设为 `127.0.0.1`。

评测器发送的内容和连接 ApxInf 服务时一样：翻转好的原始图像、状态和原始指令。缩放和清洗都在板上完成。

连接时评测器会核对服务端发布的精度和 `franka_libero` 的键名，不一致就报错断开，不会开始评测。

结果文件的每一行都记录服务端发布的契约哈希、SDK 版本，以及 `replan_steps` 等评测设置。续跑时如果这些和结果文件里记录的不同，评测器会拒绝继续，避免不同配置的结果混进同一个成功率。所以换了契约版本后要用新的结果文件。

## 这些约定是怎么来的

运行时的约定都写在[契约文件](../apx-drobotics/src/apx_drobotics/contracts/pi05_libero_w8.json)里，由 [`contract.py`](../apx-drobotics/src/apx_drobotics/contract.py) 在部署、启动和每次推理时检查。键名不在契约里，它们属于上层，和 ApxInf 一样。下面几条是实测后定下来的，修改前请先了解原因：

| 约定 | 原因 |
| --- | --- |
| 图像先由 `apx_drobotics.preprocess` 用与 `openpi_client.image_tools.resize_with_pad` 相同的算法（PIL 双线性）缩放到 224×224，再交给 SDK | SDK 自己缩放时用的是 OpenCV 的 `INTER_LINEAR`，不抗锯齿，和 OpenPI 的预处理不一致，单个像素最多差约 34/255。224 的图像 SDK 会原样使用 |
| 发送 RGB 原始数组，不用 JPEG | JPEG 有损，实测会让动作变化约 0.03 |
| 相机顺序：第三人称视角在前，手腕相机在后；第三路不用 | 官方配置 `image_mask=[true,true,false]`，第三路由运行时补黑并屏蔽 |
| 状态发 8 维（末端位置、轴角、两个手指），**但模型不使用** | 官方 LIBERO 配置没有打开 `inject_state`，HBM 也没有状态输入，改变状态不会影响输出。仍然要发送，并且由 `apx_drobotics.preprocess` 检查维度，因为运行时对维度错误不会报错 |
| 指令由 `apx_drobotics.preprocess` 清洗（去掉首尾空白，`_` 和换行替换为空格） | 运行时不清洗，`_` 会被编码成独立的 token |
| 动作是 10 步 × 7 维：前 6 维是 OSC_POSE 末端增量，第 7 维是夹爪（-1 张开，+1 合拢），直接传给 `env.step` | 在 LIBERO 中回放验证过方向和符号 |

还有一个现象：**同一服务进程里，每次推理用的随机噪声不同，但噪声序列在每次启动时都从头开始。**所以相同输入只有在"刚启动的服务、相同的请求顺序"下才会得到逐字节相同的输出。比较两次运行的结果时要注意这一点。

## 参考数据

在 RDK S600（V5.1.0，SDK v2.0.0）上实测：

| 项目 | 数值 |
| --- | --- |
| 服务启动到可用 | 约 18 秒（部署检查约 4 秒，模型加载约 14 秒） |
| 单次推理 | 约 143 ms（视觉 17、语言 66、动作 59）；新进程的第 1 次约 147 ms |
| BPU 内存 | 峰值约 5.1 GB，其中 carveout 约 4.98 GB |
| CPU 内存 | 约 290 MB（用官方示例程序测得） |

网络往返时间取决于 PC 到板子的链路，评测器会在结果里单独记录。

## 常见问题

**模型加载报 `Memory alloc failed`**：ION carveout 太小，见"准备条件"。

**服务启动时报 `CONTRACT VIOLATION`**：日志会列出每一处不一致。常见原因有：
- 改过 `pi05_libero_config.json`，比如打开了 `inject_state`。
- 模型目录里放的是 `Pi05_shoes` 等其他模型。
- 部署不完整。

重新运行部署脚本可以补上缺失的部分。脚本只看 SDK 目录和模型目录在不在，所以文件被改过时，要先删除对应的整个目录，脚本才会重新下载。

**评测器连接时报 `server precision is ...` 或 `server input fields do not match`**：连上的不是这个服务，或者服务用 `--image-keys` 等参数改过键名。

**续跑时报 `was written by a different run`**：结果文件是另一个契约版本或另一组设置写的，换一个新的 `--results-jsonl`。

**请求被拒绝，报 `S600 server rejected the request`**：消息里会写明是哪个字段不符合约定。评测器发送的内容不会触发这个错误，自己写客户端时请对照 [`preprocess.py`](../apx-drobotics/src/apx_drobotics/preprocess.py)。服务拒绝请求后继续运行，只断开这个连接。

**在 WSL 上 LIBERO 每次 reset 要半分钟左右**：WSL 的 EGL 默认通过 D3D12 转译调用 Windows 显卡，robosuite 每次 `reset` 重建 MuJoCo 渲染上下文时很慢（实测约 35 秒）。在评测命令前加 `LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe` 改用 Mesa 软件渲染，`reset` 约 3 秒。两种渲染下物理状态相同，图像只有细微差别（每像素平均差不到 1/255）。同一组结果文件里不要混用两种渲染。

**LIBERO 加载初始状态时报 `Weights only load failed`**：这是 PyTorch 2.6 以后的版本和 LIBERO 之间的问题，与 S600 无关。LIBERO 用 `torch.load` 读取初始状态文件时没有指定 `weights_only=False`。如果你信任本机的 LIBERO 数据文件，可以在评测命令前加 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。

## 相关文件

| 文件 | 作用 |
| --- | --- |
| `apx-drobotics/scripts/deploy.sh` | 板上部署 |
| `apx-drobotics/src/apx_drobotics/serving.py` | 板上的 WebSocket 服务 |
| `apx-drobotics/worker/` | C++ 推理进程，调用地瓜 SDK |
| `apx-drobotics/requirements-board.txt` | 板上的 Python 依赖 |
| `apx-drobotics/src/apx_drobotics/contracts/pi05_libero_w8.json` | 契约 |
| `apx-drobotics/src/apx_drobotics/contract.py` | 契约检查，也可以单独在命令行运行 |
| `src/apxinf_robo/engine_drobotics.py` | apxinf-robo 使用 apx-drobotics 的唯一入口 |
| `src/apxinf_robo/cli/serve.py` | 服务入口，`--engine drobotics` 参数 |
| `src/apxinf_robo/cli/eval_libero.py` | 评测器 |

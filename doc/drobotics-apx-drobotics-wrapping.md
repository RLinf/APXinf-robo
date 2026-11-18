# apx-drobotics 与 ApxInf 并行封装：代码结构、设计和命令

本文对应分支 `s600-insert-v1`。该分支先把 S600 做成与 ApxInf 并列的包（提交 `8a9b21c`、`7d95f22`），之后把引擎和包统一改名为 drobotics（`s600` 改名为 `drobotics`，`apx-s600` 改名为 `apx-drobotics`）。板子、SDK 目录 `oellm_runtime_s600` 等厂商固定名不变。

## 1 代码仓结构

下面是提交后的目录树，只列与 S600 有关的部分。`★` 表示这条 S600 线新增的文件，`△` 表示为它修改过的文件。

```text
apxinf-robo/
├── apxinf/                         ApxInf 子模块（板上不需要）
├── apx-drobotics/                    ★  独立的包：S600 上的 Pi0.5，不依赖 apxinf_robo 和 apxinf
│   ├── README.md                ★
│   ├── pyproject.toml           ★
│   ├── requirements-board.txt   ★  板上 Python 依赖的固定版本
│   ├── requirements-board-libero.txt  ★  板上进程内评测（--with-libero）要装的仿真依赖
│   ├── scripts/deploy.sh        ★  板上一键部署：SDK、模型、venv、worker、校验
│   ├── worker/                  ★  C++ 推理进程，链接官方 SDK，在板上编译
│   │   ├── CMakeLists.txt
│   │   └── pi05_worker.cc
│   ├── src/apx_drobotics/            ★
│   │   ├── policy.py               DroboticsPolicy：把前处理、运行时、后处理组合成一个策略
│   │   ├── preprocess.py           Apx-drobotics-preprocess：校验观测，缩放到 224，清洗 prompt
│   │   ├── runtime.py              启动 worker，经管道收发帧，带超时
│   │   ├── postprocess.py          Apx-drobotics-postprocess：校验动作块，整理耗时
│   │   ├── serving.py              OpenPI 兼容的 websocket 服务（对应 apxinf.serving）
│   │   ├── contract.py             契约：固定 SDK、模型文件哈希、输入输出
│   │   ├── contracts/pi05_libero_w8.json
│   │   ├── errors.py               错误分类和退出码
│   │   └── testing.py              假 SDK 和假 worker，不需要板子就能测试
│   └── tests/                   ★  包自己的测试
├── src/apxinf_robo/
│   ├── protocol.py              ★  两个引擎共用的 Policy 协议和 VIEW_SLOTS
│   ├── engine.py                △  ApxInf 入口，新增 apxinf_available()
│   ├── engine_drobotics.py           ★  S600 入口，对应 engine.py
│   ├── presets.py               △  build_robot_policy 增加 engine= 参数
│   ├── __init__.py              △  没有 ApxInf 时也能 import
│   ├── conventions/base.py、canonical.py、embodiments/base.py、embodiments/__init__.py
│   │                            △  VIEW_SLOTS 改从 protocol 取，不再 import apxinf
│   └── cli/
│       ├── serve.py             △  增加 --engine drobotics
│       └── eval_libero.py       △  增加 --engine drobotics，记录运行身份
├── README.md                    △  S600 一节：--engine drobotics 的用法
├── scripts/drobotics_pi05_*.py       ★  阶段验证用脚本（夹具、逐层追踪）
├── tests/
│   ├── test_protocol.py         ★  协议与 ApxInf 的值保持一致
│   ├── test_import_without_apxinf.py、_no_apxinf.py   ★  屏蔽 apxinf 后仍能 import
│   ├── test_serve_drobotics.py、test_eval_libero_drobotics.py   ★
└── doc/
    ├── tutorial-pi0.5-on-drobotics.md   部署和运行教程
    └── drobotics-apx-drobotics-wrapping.md   本文
```

**新增代码各自的目的**

| 新增 | 目的 |
| --- | --- |
| `apx-drobotics/` | ApxInf 不支持 BPU，S600 这条线只能用地瓜官方 SDK。把它做成独立的包，不依赖 apxinf_robo 和 apxinf。 |
| `protocol.py` | robo 的上层原来要从 ApxInf 里取 `Policy` 和 `VIEW_SLOTS`，而板上没有 ApxInf。把这两样放到 robo 自己手里，上层才能不依赖任何一个引擎。 |
| `engine_drobotics.py` | 和 `engine.py` 并列的唯一入口，robo 只经过它使用 S600。 |
| `apx-drobotics/scripts/deploy.sh` 和契约 | 板子容器可能被替换，需要能重新部署，并确认 SDK、模型、worker 与验证过的版本一致。 |
| 测试和假 SDK | 在没有板子的机器上也能验证这条路径。 |

## 2 总体设计

### 实现思路

- 整个 apxinf-robo 仓库部署在板上，客户端通过 in-process 或 websocket 把 observation 送进来、拿回 actions。
robo 负责适配机器人本体和环境（robopreset），它把观测按预设的键交给一个 `Policy`，只调用 `policy.infer(observation) -> {actions, timing}`，不关心这个策略背后是 ApxInf 还是 S600。
- S600 一侧的 `DroboticsPolicy` 内部分三步：`preprocess` 把原始观测整理成 SDK 该收到的 `ModelRequest(images, state, prompt)`；运行时把它交给官方 SDK oellm-runtime，得到 `[1, 10, 7]` 的 float64 动作；`postprocess` 校验动作并转成 `float32 [10, 7]`，再配上耗时返回。
- SDK 自带的前后处理保持开启，`preprocess` 和 `postprocess` 只补它做不到的事：按 OpenPI 的方式缩放图像、清洗 prompt、拒绝不合规的观测、拒绝含 NaN 或形状不对的动作。模型改变时，改的是这一层。

### Policy 的抽象

-  `apxinf_robo/protocol.py` 里的 `Policy`是一个 `typing.Protocol`（结构类型）。它规定了 `infer`、`metadata`、`action_dim`、`action_horizon`、`close`，以及 `infer` 的结果至少含 `actions` 和 `timing`。任何满足这个形状的对象都是策略，不需要继承。
- **没有基类，没有注册表，也没有 `ModelInput`/`ModelOutput` 之类的中间数据类型。** 两个引擎共用同一个运行时形状，区别只在加载方式，所以不需要这些。
- **选引擎用的是 `engine=` 字符串加一个 `if`**（`presets.py` 的 `build_robot_policy`），不是插件机制。以后加第三个引擎，要新增一个 `engine_<name>.py`，再在 `ENGINES` 和这处分支里各加一行。

### 调用 S600 的路径

```text
build_robot_policy("franka_libero", model_dir, engine="drobotics")          presets.py
  → engine_drobotics.load_policy(model_dir)                                  engine_drobotics.py
    → DroboticsPolicy.from_deployment(...)                                   apx_drobotics/policy.py
        1. 校验参数是否符合契约；校验部署（SDK、模型文件哈希、worker 可执行）
        2. 启动 C++ worker 子进程，等它报告就绪
        3. 组装 Preprocess、Worker、Postprocess

policy.infer(observation)
  → Preprocess            校验观测，缩放到 224，清洗 prompt，得到 ModelRequest
  → Worker.infer          经管道发给 worker 进程，带超时，同步等待
      worker 内部调用 SDK oellm-runtime，在 BPU 上跑视觉、语言、动作三段 HBM
  → Postprocess           校验动作块，转成 float32 [10, 7]，整理耗时
  → {"actions": ..., "timing": ...}
```


### robo 后端原本写死 ApxInf，现在改

原来有四处直接依赖 ApxInf，现在的处理如下：

1. **共享词汇表。** `presets.py` 和 `embodiments/base.py` 等原来从 `apxinf.policies.base` 取 `Policy` 和 `VIEW_SLOTS`。现在它们定义在 `protocol.py`，上层都改成从那里取。
2. **没有 ApxInf 时的导入。** `import apxinf_robo` 现在不需要 ApxInf。依赖 ApxInf 处理链的机器人本体（G1、YAM、双 Franka）和 `preflight` 只在 `apxinf_available()` 为真时注册和导出；没有 ApxInf 时访问它们，会得到安装提示。
3. **构建策略。** `build_robot_policy` 增加 `engine="apxinf"|"drobotics"`，默认仍是 `apxinf`，原有调用不受影响。选 `drobotics` 时，只接受没有机器人专用前后处理步骤的预设，也就是 `franka_libero`；G1、YAM、双 Franka 会直接报错。
4. **命令行。** `serve` 和 `eval-libero` 都增加 `--engine`。选 `drobotics` 时，只有 ApxInf 才支持的参数（`--checkpoint`、`--autotune`、`--tactics` 等）会报错，不会被悄悄忽略。`serve` 对 `apxinf.checkpoints` 的导入也改成了按需。

websocket 服务由各引擎自己提供：ApxInf 用 `apxinf.serving`，S600 用 `apx_drobotics.serving`，两者都经各自的 `websocket_server(...)` 入口。OpenPI 的 websocket 协议没有搬进 robo。

### 目前的限制

- 只支持 Pi05(libero) 的 W8 模型和 `franka_libero` 预设。
- 噪声由 SDK 运行时管理，不能从外部传入（传 `noise=` 会报错），所以不支持 warm start。
- 观测的键只支持一层，多余的键会被拒绝。

## 3 使用 S600 的终端指令

### 部署

```bash
bash apx-drobotics/scripts/deploy.sh                    # 默认装到 /opt/d-robotics-llm/2.0.0
bash apx-drobotics/scripts/deploy.sh --root <目录>      # 指定安装根目录
```

脚本依次做：下载 SDK 并校验、下载 Pi05 模型并校验、创建 `venv-apx-drobotics` 并装依赖、编译 worker、按契约检查部署。每一步已完成就跳过，校验不通过就停止。

### 启动服务

```bash
/opt/d-robotics-llm/2.0.0/venv-apx-drobotics/bin/apxinf-robo serve --engine drobotics --robot franka_libero \
  --model-dir /opt/d-robotics-llm/2.0.0/D-Robotics_LLM_2.0.0_S600_S100_SDK/oellm_runtime_s600 \
  --port 8000
```

日志出现 `listening on ws://...` 后才开始监听。部署检查不通过，退出码为 2；推理进程加载失败，退出码为 3。需要改 worker 路径或超时，用 `--policy-options '{"infer_deadline": 10}'`。

### websocket 评测

```bash
apxinf-robo eval-libero --backend websocket --host 127.0.0.1 --port 8000 \
  --precision w8 \
  --suite libero_10 --tasks 0 --trials-per-task 1 \
  --results-jsonl results.jsonl --summary-json summary.json
```

PC 上不需要 ApxInf。评测器连接时会核对服务端发布的精度和 `franka_libero` 的键名，不一致就断开。续跑时，如果契约哈希、SDK 版本或评测设置与结果文件里记录的不同，评测器会拒绝继续，换了契约后要用新的结果文件。

### 在板上进程内评测（in-process 评测）

把环境、评测器和推理都放在板子上，不经 websocket。部署时加 `--with-libero`：

```bash
bash apx-drobotics/scripts/deploy.sh --with-libero
```

这一步装上仿真（约 5 GB），并生成 `/opt/d-robotics-llm/2.0.0/run-libero-in-process.sh`。之后这样跑：

```bash
/opt/d-robotics-llm/2.0.0/run-libero-in-process.sh \
  --suite libero_10 --tasks 0 --trials-per-task 1 \
  --results-jsonl results.jsonl --summary-json summary.json
```

包装脚本已经设好模型目录、`--precision w8` 和渲染相关的环境变量，参数直接透传给 `eval-libero`。

不用包装脚本时，等价的直接命令是（渲染环境变量要自己设）：

```bash
MUJOCO_GL=osmesa apxinf-robo eval-libero --backend in-process --engine drobotics \
  --precision w8 --suite libero_10 --results-jsonl r.jsonl --summary-json s.json
```

`--engine drobotics` 只能和 `--backend in-process` 一起用，并且要求 `--precision w8`。

`--with-libero` 默认关闭，因为只提供推理服务用不到仿真。2026-11-10 在板上实测：libero_10 任务 0 的 60 步一轮跑通，共 12 次推理，平均 147 ms 一次（其中模型 143 ms），环境循环约 0.33 s 一步。

**板上三个不能想当然的地方**，都是实测撞出来的，改动这些设置会直接失败：

- **`MUJOCO_GL=osmesa`，不能用 `egl`。** 板上的 `libEGL` 是厂商（hobot）的构建，不导出 `eglQueryDevicesEXT`，而 mujoco 的 EGL 路径靠这个扩展枚举设备，用 `egl` 会直接报 `ImportError: eglQueryDevicesEXT is not available`。装了 `libosmesa6` 之后 OSMesa 能正常出图。
- **mujoco 必须是 3.x。** Python 3.12 上 mujoco 2.3.7 只有源码包，编译时 CMake 要从 GitHub 现场拉八个依赖仓库，板子网络撑不住（实测在 FetchContent 处四次失败）。3.3.7 有现成的 aarch64 wheel；robosuite 只要求 `mujoco>=2.3.0`。
- **torch 必须是 CPU 版。** PyPI 给 aarch64 的是 CUDA 构建，在板子上 import 会失败（`libcudart.so.13: cannot open shared object file`）。LIBERO 只用 `torch.load` 读初始状态，CPU 版足够。部署脚本从 `https://download.pytorch.org/whl/cpu` 装它。

另外两处与 LIBERO 自身有关的处理，已固化进部署脚本：它上游仓库里的 `libero/__init__.py` 没有被 git 跟踪，新克隆出来是没有 `libero` 包的，脚本会补一个空文件；它的 asset 路径来自 `$LIBERO_CONFIG_PATH/config.yaml`，文件缺失时它会从 stdin 读入而卡死，脚本会预先写好这份配置。

`requirements-board-libero.txt` 里锁死了这一整套版本。它和 `requirements-board.txt` 一起装：后者以约束文件的形式传入，保证 numpy 1.26.4 和 pillow 12.3.0 不被改动，因为 letterbox 缩放依赖这两者的字节级一致。

### 关于结果文件

websocket 和 in-process 可以写同一个 `--results-jsonl`，账本不区分传输方式。但要注意：两种方式渲染观测的不是同一套 GL 栈——websocket 那条路的观测产生在客户端机器上（通常是 PC），in-process 产生在板子上。两边缩放和抗锯齿的细节可能不同，同一任务的观测不会逐字节一致。要让两次结果可比，就分开写文件。

### 在 Python 里直接使用

```python
from apxinf_robo import build_robot_policy

policy = build_robot_policy("franka_libero", None, engine="drobotics")   # None 表示用部署脚本的默认位置
result = policy.infer({
    "observation/image": base,          # HWC uint8 RGB，任意尺寸
    "observation/wrist_image": wrist,
    "observation/state": state,         # float32，8 维
    "prompt": "pick up the book",
})
result["actions"]                       # float32 [10, 7]
policy.close()
```

### 测试（不需要板子）

```bash
pip install -e './apx-drobotics[serve,dev]'
(cd apx-drobotics && pytest)                 # apx-drobotics 自己的测试
pytest tests/test_protocol.py tests/test_import_without_apxinf.py \
       tests/test_serve_drobotics.py tests/test_eval_libero_drobotics.py   # robo 一侧
```


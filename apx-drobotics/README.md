# apx-drobotics

把 D-Robotics S600 上的 π0.5 封装成与 ApxInf 策略相同的调用形式，使它和 ApxInf 并列，由 apxinf-robo 统一使用。

```text
obs / action
    │  in-process，或 websocket（apxinf-robo serve）
apxinf-robo：eval-libero、franka_libero、build_robot_policy
    ├─ engine.py ──────→ ApxInf
    └─ engine_drobotics.py ─→ apx_drobotics
                           ├ preprocess.py    原始观测 → SDK 该收到的内容
                           ├ runtime.py       C++ worker → SDK oellm-runtime（黑盒）
                           └ postprocess.py   SDK 的回复 → {actions, timing}
```

这个包不依赖 ApxInf 和 apxinf-robo，后者通过 `DroboticsPolicy` 的 `infer(observation) -> {actions, timing}` 使用它。目前只支持 `Pi05(libero)` W8 模型和 `franka_libero` 预设。

## 目录

| 位置 | 内容 |
| --- | --- |
| `src/apx_drobotics/policy.py` | `DroboticsPolicy`，把三层组合成一个策略 |
| `src/apx_drobotics/preprocess.py` | 校验观测，缩放到 224，清洗 prompt |
| `src/apx_drobotics/runtime.py` | 启动 worker，走管道收发帧，带超时 |
| `src/apx_drobotics/postprocess.py` | 校验动作块，整理耗时 |
| `src/apx_drobotics/serving.py` | OpenPI 兼容的 websocket 服务，对应 `apxinf.serving` |
| `src/apx_drobotics/contract.py`、`contracts/` | 契约：固定 SDK、模型、运行时配置和输入输出 |
| `src/apx_drobotics/testing.py` | 假 SDK 和假 worker，不需要板子就能测试 |
| `worker/` | C++ worker，在板上编译 |
| `scripts/deploy.sh` | 板上部署脚本 |

## 为什么 SDK 外面还要加一层

SDK 自带的前后处理保持开启，这里只负责它做不到的事：

- **缩放**：SDK 只在图像不是 224 时才缩放，用的是不抗锯齿的 `cv2.INTER_LINEAR`，和 OpenPI 的预处理不一致，单个像素最多差约 34/255。所以先按 OpenPI 的方式缩放到 224，SDK 会原样使用。
- **prompt**：SDK 不清洗 prompt，下划线会变成单独的 token。ApxInf 的 tokenizer 会清洗，这条路径上就由 preprocess 来做。
- **校验**：SDK 收到 7 维状态或没清洗的 prompt 都不会报错，只有成功率下降。所以不合规的观测直接拒绝，并列出所有问题。

## 用法

部署和运行见 [S600 教程](../doc/tutorial-pi0.5-on-drobotics.md)。在 Python 里直接使用：

```python
from apxinf_robo import build_robot_policy

policy = build_robot_policy("franka_libero", None, engine="drobotics")
result = policy.infer({
    "observation/image": base,             # HWC uint8 RGB，任意尺寸
    "observation/wrist_image": wrist,
    "observation/state": state,            # float32，8 维
    "prompt": "pick up the book",
})
result["actions"]                          # float32 [10, 7]
policy.close()
```

错误分三类：`RequestError` 表示观测不合规，策略还能继续用；`DeploymentError` 和 `StartupFailure` 表示没能启动；`RuntimeFailure` 表示运行时出错，策略会停止服务。出错后不会自动重启 worker，因为这会掩盖故障，还会让运行时的噪声序列从头开始。

## 与 ApxInf 的差别

- 只有 `metadata`、`infer`、`close`、`action_dim`、`action_horizon` 这些共同接口，没有 ApxInf 的 L1 接口。
- 噪声由运行时管理，每个进程内的序列固定，每次推理前进一步，不能从外部传入，所以不支持 warm start。
- 输入按契约严格校验，多余的键也会被拒绝。观测的键只支持一层，不支持 `images/cam_high` 这样的路径。
- 不支持带机器人专用处理步骤的预设（G1、YAM、双 Franka），这些步骤运行在 ApxInf 的处理链里。

## 测试

不需要板子，在 PC 上运行：

```bash
pip install -e './apx-drobotics[serve,dev]'
cd apx-drobotics && pytest
```

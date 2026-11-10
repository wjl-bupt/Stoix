# cwq/checks — 当前代码的正确性检查

每个文件是一次实现改动时写的独立检查（数值恒等式、梯度等价、规范中性、开关默认关时逐位不变等），全部可以在 CPU 上跑：

```bash
cd /home/junweiluo/Stoix && JAX_PLATFORMS=cpu .venv/bin/python cwq/checks/<file>.py
```
最后一行打印 `k/n passed`。部分检查读取已有存档（`/home/junweiluo/stoix_runs/minatar_g8ck/ckpt`、`breakout_miss/bank.pkl`）。

| 文件 | 检查对象 |
|---|---|
| check_same_traj.py / check_rho.py / check_rho_sparse.py | 同轨迹分组采样、ρ 配对核（行和、对称、规范中性） |
| check_rpair.py / check_c51.py | 成对 critic 各头的公式与 C51 投影 |
| check_qcw.py | MinAtar CW-Q（head=qcw）与理论式 ξ = v_j u_i − v_i u_j 一致 |
| check_qcw_nstep.py | n-step CW-Q-ρ（nstep_tail 与暴力实现一致、n=1 不变） |
| check_qcw_nonorm.py | MinAtar nonorm：梯度 = 原版 × (1−γ)、规范中性 |
| check_qcw_cont.py / check_cmin_cont.py | 连续动作 CW-Q 与去水平 min |
| check_no_norm_cont.py | 连续动作 nonorm（cw_continuous 与 Gym 脚本）：默认逐位不变、梯度 = 原版 × (1−γ) |
| check_mujoco.py | Gym MuJoCo 单文件实现 |

其余历史检查（针对已放弃的变体）在 `archive/scripts_rdqn/check_*.py`。

# cwq/ — Relative-Q（CW-Q-ρ）项目入口

本目录是 Stoix `rdqn` 分支上 relative-Q 项目的**唯一实验入口**：每个 benchmark 一个启动脚本，脚本里同时包含我们的
CW-Q-ρ 系列算法和该 benchmark 的基线，默认参数就是当前实验协议。算法实现仍在 `stoix/` 下（见"代码在哪里"）。
2026-11-10 之前的一次性脚本（252 个 launch/stop/patch/check/smoke…）已整体归档到 `archive/`，见 `archive/README.md`。

```
cwq/
├── README.md            本文件
├── run_minatar.sh       MinAtar（pgx）           CW-Q-ρ / CW-Duel-ρ + DQN、DDQN、Dueling、C51、R-C51、RC（PQN 可选）
├── run_dmc.sh           DeepMind Control（MJX）   CW-Q-ρ(TD3 去水平 min / TD3 原 min / DDPG) + TD3、DDPG、SAC
├── run_mujoco.sh        Gym MuJoCo v5            同上（单文件实现 mujoco_gym/mujoco_offpolicy.py）
├── status.sh            查看某个结果目录的进度        bash cwq/status.sh <OUT>
├── stop.sh              停止某个结果目录的启动器及其全部 run   bash cwq/stop.sh <OUT>
├── lib/common.sh        三个启动器共用的队列/重试/GPU 逻辑
├── lib/gpu_busy.sh      与占卡程序的约定（跑实验时让出 GPU）
├── mujoco_gym/          Gym MuJoCo 的单文件 JAX 实现（TD3/DDPG/SAC + CW-Q）
├── analysis/plot_curves.py   任意一组 run 的学习曲线（均值±标准误）+ AUC/final 表
├── checks/              当前代码的正确性检查（梯度恒等式、规范中性等），见 checks/README.md
└── tools/               GPU 占卡程序、旧结果导入工具
```

## 快速开始

所有命令在仓库根目录 `/home/junweiluo/Stoix` 下执行。启动器默认**自己转到后台**（setsid+nohup）并立刻返回，
打印结果目录和查看/停止命令；`FG=1` 则在前台跑。

```bash
DRY_RUN=1 bash cwq/run_minatar.sh | tail        # 只打印将要执行的命令和数量，不启动任何东西（建议每次先看一眼）
bash cwq/run_minatar.sh                          # 默认：CW-Q-ρ + 6 个基线 × 5 游戏 × 10 seeds × 5M
bash cwq/status.sh /home/junweiluo/stoix_runs/minatar_main
bash cwq/stop.sh   /home/junweiluo/stoix_runs/minatar_main
```

常见用法（参数都是环境变量，写在命令前面）：

```bash
ALGOS="cwq" GAMES="asterix breakout" SEEDS="0 1 2 3 4" bash cwq/run_minatar.sh    # 只跑我们的方法、两个游戏、5 seeds
ALGOS="cwq" RHO=0.0 bash cwq/run_minatar.sh        # 改 ρ：run 名自动变成 cwq_rho0.0，不会覆盖默认版
ALGOS="cwq" NONORM=0 bash cwq/run_minatar.sh       # 旧版（ξ/(1−γ)）：run 名 cwq_norm
ALGOS="cwq" NSTEP=3 G=2 bash cwq/run_minatar.sh    # 组合：cwq_g2_n3
ALGOS="cwq dqn" LR=1e-4 TAG=lr1e4 bash cwq/run_minatar.sh   # 改协议参数必须给 TAG（所有算法的 run 名都加 _lr1e4）
ALGOS="cwq" EXTRA_OURS="+system.rho_term_shuffle=True" TAG=tshuf bash cwq/run_minatar.sh  # 任意 Hydra 覆盖，同样必须给 TAG
ALGOS="cwq_td3 cwq_ddpg" TASKS="cheetah_run hopper_hop" bash cwq/run_dmc.sh
ALGOS="cwq_td3" ENVS="Hopper-v5 Walker2d-v5" bash cwq/run_mujoco.sh
```

同一个结果目录可以反复启动：已经结束（jsonl 最后一行含 `"done"`）的 run 会被跳过，没结束的会重跑；
同一目录同时只允许一个启动器（`_launcher.pid`）。不同变体靠 run 名区分，所以默认每个 benchmark 只有一个结果目录
（`minatar_main` / `dmc_main` / `mujoco_main`），需要隔离时可以用 `OUT=...` 另起目录。

## 算法与 run 名

| benchmark | ALGOS 名 | 是什么 | 旧实验里的名字 |
|---|---|---|---|
| MinAtar | `cwq` | **CW-Q-ρ（主方法）**：普通 Q 头，全配对交叉加权成对损失，同轨迹 G 个样本一组的 ρ 配对核，nonorm | `qcw_nonorm`（nonorm）/ `qcw_rho`（旧版） |
| MinAtar | `cwq_duel` | CW-Duel-ρ：dueling 头（rank 0），同一损失和配对核 | `cwduel_rho_r0`（旧版） |
| MinAtar | `dqn ddqn dueling c51 rc` | 社区 DQN 系基线（`ff_rdqn.py`） | 同名 |
| MinAtar | `rc51` | R-C51（Bellman uniformization，`ff_rpair.py head=rc51`） | 同名 |
| MinAtar | `pqn` | PQN（λ 回报，协议不同，不在默认基线里） | 同名 |
| DMC / Gym | `cwq_td3` | **CW-Q-ρ(TD3)，去水平 min**（两个 critic 先减去各自的批均值再取 min） | `cwq_cmin` |
| DMC / Gym | `cwq_td3raw` | CW-Q-ρ(TD3)，原始 min（会被两个 critic 的水平差劫持，几乎总选 critic 0） | `cwq_rho` |
| DMC / Gym | `cwq_ddpg` | CW-Q-ρ(DDPG) | `cwq_ddpg` |
| DMC / Gym | `td3 ddpg sac` | 基线 | 同名 |

run 名 = 算法名 + 我们方法的非默认参数后缀 + `_TAG`：`_rho<ρ>`、`_g<G>`、`_norm`（NONORM=0）、`_n<NSTEP>`、`_k<RHO_CROSS_K>`。
基线不受这些参数影响，所以只加 `_TAG`。结果文件是 `<OUT>/<游戏或任务>_<run 名>_s<seed>.jsonl`
（DMC 的任务名取 env yaml 里的 task_name，例如 `WalkerWalk`）。

**nonorm（2026-11-10 起默认）**：CW 残差不再除以 1−γ，即目标用 sg[D+ξ] 而不是 sg[D+ξ/(1−γ)]。它在数学上等于把整个梯度
乘以 1−γ；因为 Adam 对整体尺度不变（梯度裁剪从未触发），实际效果等价于"旧版 + Adam eps 放大 1/(1−γ)=100 倍"
（Adam(c·g, eps) = Adam(g, eps/c)）。MinAtar 上已验证（asterix +6.3 AUC、SI +11.7，其余持平）；DMC 和 Gym MuJoCo 上
**还没有跑过 nonorm**，启动器默认也用 nonorm，旧结果对应 `NONORM=0`（`_norm`）。Gym MuJoCo 的 Adam eps 是 optax 默认
1e-8，那里 nonorm 的作用预计很小。

## 默认协议

**MinAtar**（`run_minatar.sh`）：pgx MinAtar（sticky 0.1，单局上限 10000 步），asterix / breakout / freeway / seaquest /
space_invaders，seeds 0–9，5M 环境步；128 个并行环境、每个向量步 8 次梯度更新（epochs 8）、batch 512、回放 100k、
Adam lr 5e-4（eps 1e-5）、梯度裁剪 10、ε 1→0.1（100k 步）、目标网络每 1000 次更新、warmup 40 个向量步、γ=0.99；
50 次贪心评估 × 128 局；在 20/40/60/80/100% 存网络（`CKPT=1`）。我们的方法：ρ=0.5、G=8、nonorm。

**DMC**（`run_dmc.sh`）：MuJoCo Playground 官方 brax_sac_config（128 环境、UTD 1/16、batch 512、回放 4M、lr 1e-3、
Q 网络 LayerNorm、256×256 ReLU、warmup 64 个向量步随机动作、截断自举），TD3 机制（探索噪声 .1、目标噪声 .2/.5、
策略延迟 2）；**不做观测归一化**（off-policy 不加归一化 wrapper）。6 个任务（walker_walk、walker_run、hopper_hop、
finger_spin、reacher_hard、cheetah_run），seeds 0–9，5M 步（walker_run/hopper/finger/cheetah 10M），50 次评估 × 32 局。
DMC 没有终止，交叉加权不起作用，CW-Q-ρ 在这里等于"批内去水平的 TD + ρ 配对核"。每张卡只放 1 个 run（多个 MJX
进程共卡会触发驱动段错误），每个 run 需要约 8.6GB 显存（`MINFREE=10000` 等显存够了才启动）。cheetah_run 自动加
`+env.batch_cond_reset=True +system.no_batch_vmap=True`（自动重置的 200 步稳定化原来每步都算，修复后快约 85 倍）。

**Gym MuJoCo**（`run_mujoco.sh`）：CleanRL 默认配置（1 个环境、1M 步、UTD 1、batch 256、回放 1M、256×256、
TD3/DDPG lr 3e-4、25k 随机 warmup；SAC q-lr 1e-3、π-lr 3e-4、自动 α），截断自举、只有真正失败才算终止；
Humanoid / Ant / Walker2d / Hopper / HalfCheetah，seeds 0–4；每 5000 步评估 10 局。

## 参数一览

| 变量 | MinAtar 默认 | DMC 默认 | Gym 默认 | 说明 |
|---|---|---|---|---|
| `ALGOS` | `cwq dqn ddqn dueling c51 rc51 rc` | `cwq_td3 cwq_ddpg td3 ddpg sac` | 同 DMC | 要跑的算法 |
| `GAMES` / `TASKS` / `ENVS` | 5 个游戏 | 6 个任务 | 5 个环境 | |
| `SEEDS` | `0 … 9` | `0 … 9` | `0 … 4` | |
| `RHO` `G` `NONORM` | 0.5 8 1 | 0.5 8 1 | 0.5 8 1 | 只作用于我们的方法，非默认值进 run 名 |
| `NSTEP` `RHO_CROSS_K` | 1 0 | – | – | MinAtar 专有 |
| `EXTRA_OURS` / `EXTRA_ALL` | 空 | 空 | 空 | 原样追加的覆盖参数；非空时必须给 `TAG` |
| 协议参数 | `TT=5e6 NUM_EVAL=50 EPOCHS=8 LR=5e-4 BATCH=512 CKPT=1` | `STEPS`（覆盖步数表） | `TOTAL=1000000 CKPT=0` | 改了就必须给 `TAG` |
| `TAG` | 空 | 空 | 空 | 追加到所有 run 名 |
| `GPU_LIST` | `0 1 2` | `0 1 2` | `0 1 2` | |
| `NW_PER_GPU` | 3 | 1 | 8 | 每张卡同时跑几个 |
| `MINFREE` | 0 | 10000 | 0 | 启动前等待的空闲显存（MB） |
| `RETRIES` | 30 | 30 | 3 | 每个 run 最多尝试次数（启动时段错误/OOM 很常见） |
| `OUT` | `$RUNS_ROOT/minatar_main` | `$RUNS_ROOT/dmc_main` | `$RUNS_ROOT/mujoco_main` | `RUNS_ROOT=/home/junweiluo/stoix_runs` |
| `DRY_RUN` `FG` | 0 0 | | | 只打印 / 前台运行 |

## 结果目录里有什么

`<OUT>/*.jsonl` 每个 run 一个（第一行是完整 config，之后每次评估一行，最后一行 `{"done": true, ...}`）；
`_logs/<run>.a<k>.log` 第 k 次尝试的输出；`_logs/progress.txt` 每次尝试的 DONE/FAIL/耗时；`_logs/launcher.out`；
`_code/` 启动时的启动器和算法源码快照；`STARTED`（启动时间和参数）、`ALLDONE_<时间>`、`STOPPED_<时间>`；
`ckpt/` 存档；`_imported.txt` 从旧目录导入的 run 清单。

已经把旧实验中**与当前协议逐项一致**的 run 导入了三个 main 目录（见下节），所以按默认参数启动只会补缺的 run。

## 旧结果导入（tools/import_legacy.py）

```bash
DRY_RUN=1 ALGOS="dqn" bash cwq/run_minatar.sh > /tmp/dry.txt
.venv/bin/python cwq/tools/import_legacy.py /tmp/dry.txt dqn=/home/junweiluo/stoix_runs/minatar_pgx:dqn           # 只检查
.venv/bin/python cwq/tools/import_legacy.py /tmp/dry.txt dqn=/home/junweiluo/stoix_runs/minatar_pgx:dqn --apply   # 复制
```
它把启动器会用的每一个覆盖参数和旧 run 的 config 逐项比较，全部一致且旧 run 已完成才复制（只复制，不移动，不覆盖）。

## 代码在哪里

| 文件 | 内容 |
|---|---|
| `stoix/systems/q_learning/ff_rpair.py` | MinAtar 成对方法：CW-Q-ρ（head=qcw）、CW-Duel（head=duel）、CW-Pair（head=free/anti/lowrank）、R-C51（head=rc51）；`duel_all_pairs` 是 CW-Q-ρ 的全配对损失；各种实验开关（cw_no_norm、n_step、rho_cross_k、rho_term_shuffle、qcw_full_target…） |
| `stoix/systems/q_learning/same_traj.py` | 同轨迹分组采样（G 个一组）与 ρ 配对核（rho_weight_matrix 及其变体） |
| `stoix/systems/q_learning/ff_rdqn.py` | DQN/DDQN/Dueling/C51/Reward centering/RDQN（MinAtar 基线） |
| `stoix/systems/q_learning/ff_pqn_m.py` | PQN（MinAtar） |
| `stoix/systems/ddpg/cw_continuous.py` | 连续动作的全部逻辑：TD3/DDPG/SAC 底座 + head=q/qcw/pair/duel，min_mode，no_norm |
| `stoix/systems/ddpg/ff_cw_{td3,ddpg,sac}.py` | 上面的入口；`keepalive.py` 是驱动段错误的规避（可执行文件永不卸载） |
| `stoix/utils/make_env.py` | pgx MinAtar 与 MJX 环境构造（含 BatchCondAutoReset） |
| `stoix/configs/env/pgx/`、`configs/env/mjc_playground/dm_control/` | 环境 yaml |
| `cwq/mujoco_gym/mujoco_offpolicy.py` | Gym MuJoCo 单文件实现（与 cw_continuous 的 qcw 公式相同） |

## GPU 占卡（tools/）

`tools/gpu_hold_mgr.sh`：对 `GPUS` 中每张卡，没有我们的实验在用时占住 `TARGET_MB`（别人释放显存后自动加占，从不碰别人的进程）；
启动器在每个 worker 开始时通过 `lib/gpu_busy.sh` 打"忙"标志并杀掉该卡上的占位进程，结束后去掉标志、管理器再占回来。
- 改卡/改目标（不丢已占的显存）：`GPUS="0 1 2" TARGET_MB=23552 bash cwq/tools/gpu_hold_restart.sh`
- 全部停掉并释放：`bash cwq/tools/gpu_hold_stop.sh`
- 日志与状态文件：`/home/junweiluo/gpu_busy/`（log、mgr.pid、holder_<gpu>_<n>.pid/.mb、busy_<gpu>_<pid>）

## 坑（都已在启动器里处理）

- XLA GPU autotune 会段错误：MinAtar/Gym 用 `--xla_gpu_autotune_level=0`，DMC 另加三项编译并行/command buffer 关闭；
  根因是驱动 libcuda 580 的堆损坏，容器内无法根治，所以靠重试（MinAtar/DMC 30 次）。
- 启动时 OOM（卡上只剩 ~3GB）很常见，同样靠重试排队；失败的 jsonl 会被删掉重跑。
- 不要在 ssh 命令行里写会被 `pkill -f`/`pgrep -f` 匹配到的文本，否则会杀掉自己的 shell；停止一律用 `cwq/stop.sh`。

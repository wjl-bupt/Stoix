# archive/ — 2026-11-10 之前的脚本归档

2026-11-10 整理仓库时，`scripts_rdqn/`、`scripts_mujoco/`、`scripts_rppo/` 原样移到这里（文件内容未改，脚本里的路径
仍指向旧位置，所以**不要直接运行**，需要复现旧实验时先改路径或用 `cwq/` 的启动器）。新的入口见 `cwq/README.md`。

| 目录 | 内容 |
|---|---|
| scripts_rdqn/ | MinAtar / DMC / Kinetix 的全部旧启动器、补丁、检查、诊断与分析脚本（10-30 至 11-10） |
| scripts_rdqn/ranking/ | 同状态排序评测、Breakout 漏球、κ 噪声、E1/E2、估计量诊断、线性探针等分析脚本 |
| scripts_mujoco/ | Gym MuJoCo 旧启动器（实现已移到 cwq/mujoco_gym/mujoco_offpolicy.py） |
| scripts_rppo/ | relative PPO（rppo）实验脚本 |
| code_bak/ | stoix/ 与 scripts 下所有 *.bak_* 源码备份，按原相对路径存放 |
| patches_20261110/ | 本次整理时应用的补丁（连续动作 nonorm） |

## scripts_rdqn/

### 启动器（每个对应一次实验；OUT=结果目录，位于 /home/junweiluo/stoix_runs/）

| 文件 | 结果目录 | 说明 |
|---|---|---|
| launch_al_minatar.sh | minatar_al | 2026-11-09 (user): advantage learning 2x2 -- CW-Q-rho G=8 + AL (beta=0.9, 5 games x s0-4) and DQN + AL (breakout/freeway/space_invaders x s0-4; asterix/seaquest |
| launch_cfpair.sh | cfpair | CW-Pair MLP K=8 + counterfactual same-state pairs (2026-11-02): arms crn / ind, breakout + asterix, seeds 0-4, 5M real steps. |
| launch_crn_ctrl.sh | minatar_crn_ctrl | 2026-11-09 (user): controls for the CW-Q-rho+CRN failure (same protocol as launch_qcw_crn.sh): rr16 = no CRN (only the doubled replay per real step), crn_nomix  |
| launch_cw_closed.sh | minatar_cw_closed | 2026-11-09 (user): 2x2 factorial of the closed-form CW-rho (system.cw_closed A/B/C/D: E2 trajectory weight x E1 trajectory level), breakout/asterix; A (== CW-rh |
| launch_cw_kernel.sh | minatar_cw_kernel | 2026-11-09 (user): GAUGE-NEUTRAL E1/E2 factorial via symmetric kernels: ker_B2 (E1 on, E2 off; cw_kernel=B2), ker_C2 (E1 off, E2 on; cw_kernel=C2), rho1 (both o |
| launch_cwduel_A.sh | cwduel_A | CW-Duel stage A, 2026-11-02: asterix/seaquest x {cwduel_r32, cwduel_r0} x seeds 0-2 x 5M. |
| launch_cwduel_A2.sh | cwduel_A2 | CW-Duel stage A v2 (2026-11-02 14:30): current-state parts of Amax/Bmin from the TARGET network (duel_cur_target=True); the online-sg variant (cwduel_A) diverge |
| launch_cwduel_full.sh | cwduel_full | CW-Duel full MinAtar run (2026-11-02 15:30): head=duel_mlp (antisymmetric MLP state-pair head + centred advantage), |
| launch_cwduel_full_lr.sh | cwduel_full_lr | CW-Duel full MinAtar run, LOW-RANK version (2026-11-02 16:10): head=duel, all m(m-1) pairs, pair_rank=32, |
| launch_diag.sh | diag_gap | DIAG 2026-11-01: asterix/seaquest x {dqn, cwpair_mlp, cwpair_lr} x seeds 0-2 x 5M, same protocol as the main pgx run |
| launch_diag_one.sh | diag_gap | DIAG 2026-11-01: asterix/seaquest x {dqn, cwpair_mlp, cwpair_lr} x seeds 0-2 x 5M, same protocol as the main pgx run |
| launch_dmc.sh | dmc_cwq | 2026-11-06 v9 DMC launcher: TD3 / CW-Q-rho / DDPG / SAC (cw_continuous.py), protocol dmc_pg_common.sh, seeds 0-9. |
| launch_dmc_v10.sh | dmc_cwq | 2026-11-08 v10 DMC launcher = v9 with the holder rule changed (user): no holder while our job runs on the GPU; |
| launch_dmc_v2.sh | dmc_cwq | 2026-11-05 DMC main run: TD3 vs CW-Q-rho (cw_continuous.py head=q / head=qcw), protocol dmc_pg_common.sh |
| launch_dqn_st.sh | minatar_dqn_st | 2026-11-08 DQN with the rho grouped same-trajectory sampler (st_group 8; DQN loss unchanged, no pairs/rho). |
| launch_duelsa.sh | minatar_duelsa_ep8 | 2026-11-04 CW-Duel with pairwise (s,a) advantage head (head=duel_sa), rho=.5, OLD protocol epochs 8 (same as minatar_rho_ep8). |
| launch_ep8.sh | minatar_ep8_more | 2026-11-04 OLD protocol (epochs 8, batch 512, same as minatar_pgx / minatar_rho_ep8) additional arms, 5 games x seeds 0-9 x 5M. |
| launch_ep8b.sh | minatar_ep8_more | 2026-11-04 OLD protocol (epochs 8, batch 512, same as minatar_pgx / minatar_rho_ep8) additional arms, 5 games x seeds 0-9 x 5M. |
| launch_ep8c.sh | minatar_ep8_more | 2026-11-04 OLD protocol (epochs 8, batch 512, same as minatar_pgx / minatar_rho_ep8) additional arms, 5 games x seeds 0-9 x 5M. |
| launch_fair.sh | phase1_fair | Supplement to stage 1: budget-matched CRN.  CRN at 2.5M real steps = 5M env calls, epochs 32 (2x), batch 256 |
| launch_fulltgt.sh | minatar_fulltgt | 2026-11-10 (user): CW-Q-rho G=8 rho=.5 with the whole stop-grad target from the target net (qcw_full_target); asterix+breakout x s0-4, main protocol, ckpts. |
| launch_fw_reward.sh |  | Freeway re-run with rollout_reward_per_step logged (training-reward curve); same configs as phase1 / phase1_fair. |
| launch_g16_minatar.sh | minatar_g16 | 2026-11-08 (user): CW-Q-rho(DQN) with st_group 16 (qcw_rho_g16, 5 games x s0-9) + DQN rerun with checkpoints |
| launch_g16_mujoco.sh | mujoco_g16 | 2026-11-08 (user): CW-Q-rho(DDPG) with G 16 (cwq_ddpg_g16 = --base ddpg --head qcw --sampler rho --rho .5 --G 16, |
| launch_g2_minatar.sh | minatar_g2 | 2026-11-09 (user): CW-Q-rho with st_group=2 (one same-trajectory partner per item), 5 games x s0-9, main protocol (epochs 8, batch 512, 5M), checkpoints; else i |
| launch_g8ck_minatar.sh | minatar_g8ck | 2026-11-09 (user): G=8 CW-Q-rho rerun WITH checkpoints (qcw_rho_g8ck, s0-2) for the G=8 vs G=16 ranking comparison; else identical to launch_g16_minatar.sh |
| launch_kinetix.sh | kinetix_off | 2026-11-08 Kinetix off-policy main run (kinetix_pg_common.sh protocol): arms td3 cwq_cmin ddpg cwq_ddpg sac x seeds 0-2, |
| launch_minatar_base.sh | minatar_base | MinAtar-protocol baselines (2026-10-31): {dqn, rdqn, ddqn, dueling} x {asterix, breakout, freeway, space_invaders} |
| launch_minatar_pgx.sh | minatar_pgx | Main MinAtar benchmark (2026-10-31): pgx MinAtar (5 games, sticky 0.1, episode cap 10000) x 9 algorithms x seeds 0-9 x 5M. |
| launch_minatar_rdqn_a.sh | minatar_base | MinAtar-protocol baselines (2026-10-31): {dqn, rdqn, ddqn, dueling} x {asterix, breakout, freeway, space_invaders} |
| launch_mlpall.sh | minatar_pgx | CW-Pair, MLP free head + ALL pairs (additive decomposition), 2026-11-01. Same protocol as launch_minatar_pgx.sh. |
| launch_nonorm_g05.sh | minatar_nonorm_g05 | 2026-11-10 (user): CW-Q-rho G=8 rho=.5: (1) no 1/(1-gamma) normalisation (cw_no_norm), (2) gamma=0.5 with normalisation; asterix+breakout x s0-4, main protocol, |
| launch_nonorm_rest.sh | minatar_nonorm_g05 | 2026-11-10 (user): CW-Q-rho nonorm (new default) on the remaining MinAtar games freeway/seaquest/space_invaders x s0-4, main protocol, ckpts. Same dir as asteri |
| launch_nonterm.sh | nonterm | CW-Pair MLP K=8 trained ONLY on pairs whose two transitions both did not terminate (2026-11-02, +system.nonterm_only=True). |
| launch_nstep.sh | minatar_nstep | n-step RDQN (2026-10-31): {dqn3, rdqn3, sep3} x 4 MinAtar x seeds 0-4 x 5M, + dqn1traj (protocol check, seeds 0-2). |
| launch_phase1.sh | phase1 | Stage 1 (2026-10-30): {DQN, RDQN} x {plain, CRN} x {asterix, freeway} x seeds {0,1,2}, 5M real env steps each. |
| launch_qcw_crn.sh | minatar_qcw_crn | 2026-11-09 (user): CW-Q-rho + same-state different-action pairs from the CRN counterfactual branch (system.crn=True, cf_same, same_coef 1). |
| launch_qcw_nstep.sh | minatar_qcw_nstep | 2026-11-09 (user): 3-step CW-Q-rho (G=8 dense rho=0.5, qcw all pairs; same_traj.nstep_tail + per-item discount), lr 5e-4 (protocol default) and 1e-4, 5 games x  |
| launch_qcw_smc.sh | minatar_qcw_smc | 2026-11-09 (user): CW-Q-rho G=8 with qcw output Q - E_softmax(Q)[Q] (system.qcw_center=softmax), 5 games x s0-4, main protocol, checkpoints |
| launch_qcw_smcsg.sh | minatar_qcw_smcsg | 2026-11-09 (user): CW-Q-rho G=8 with qcw output Q - E_softmax(Q)[Q] (system.qcw_center=softmax_sg: weights stop-gradient), 5 games x s0-4, main protocol, checkp |
| launch_rc51_rho.sh | minatar_rc51_rho | 2026-11-09 (user): R-C51 + rho pairing (same_traj G=8, rho=0.5, rho_kc=7 cross partners; = CW-Pair-rho pair structure), breakout + asterix x s0-4, main protocol |
| launch_reset.sh | minatar_reset | Reset-continuing (2026-10-31): termination -> transition to the reset state, done stored as 0. |
| launch_rho01.sh | minatar_rho01 | 2026-11-10 (user): CW-Q-rho G=8 with pair_rho=0.0 and 1.0, asterix+breakout x s0-4, main protocol (= qcw_rho in ep8_more / tshuf), checkpoints. |
| launch_rho_k7_minatar.sh | minatar_rho_k7 | 2026-11-09 (user): CW-Q-rho G=8, cross-trajectory rho shared by K=G-1=7 partners per item (same_traj.rho_weight_matrix_sparse), 5 games x s0-9, main protocol (e |
| launch_rho_td.sh | minatar_rho_td | 2026-11-09 (user): intervention for M1/M2 = terminal-decoupled rho (qcw_rho_td: G=8 main config + rho_term_decouple), breakout/seaquest/asterix s0-4, with check |
| launch_rpair.sh | minatar_pair | Pairwise mean critic (ff_rpair.py, 2026-10-31): head {anti, free} x term {cross, unif} x 4 MinAtar x seeds 0-4 x 5M. |
| launch_rpair2.sh | minatar_pair2 | Round 2 (2026-10-31): same-capacity DQN (sadqn) x 4 games; diagnostics breakout K=8 partners, freeway reset/K=8. |
| launch_sametraj.sh | same_traj | 2026-11-03: same-trajectory pairing.  pgx MinAtar asterix + seaquest x seeds 0-4 x 5M, protocol = minatar_pgx. |
| launch_svc.sh | strat_vs_cw | 2026-11-03: original CW-Pair MLP K=8 vs terminal-stratified CW-Pair (strat_term=32), nothing else stacked. |
| launch_tshuf.sh | minatar_tshuf | 2026-11-10 (user): E1 causal control -- CW-Q-rho G=8 with terminal same-trajectory pair weights moved to another trajectory (system.rho_term_shuffle, same_traj. |
| launch_u32.sh | minatar_u32 | 2026-11-03 MAIN MinAtar benchmark, community update count: 5M env steps -> 1.25M gradient updates (replay ratio 1/4). |
| launch_vr.sh | vr | CW-Pair MLP K=8 + unbiased variance reduction for terminal pairs (2026-11-02): |

### 启动器包装

| 文件 | 说明 |
|---|---|
| start_duelsa.sh | 2026-11-04 start head=duel_sa experiment; hand over GPU0/1 guard from the minatar_rho_ep8 run. |
| start_ep8_more.sh |  |

### 批处理/分析驱动

| 文件 | 说明 |
|---|---|
| run_d3.sh | usage: bash run_d3.sh <gpu> <worker> <n_workers> |
| run_duelsa_guarded.sh | 2026-11-04: runs launch_duelsa.sh on GPU0/1 and keeps the guard alive until the orphaned minatar_rho_ep8 launcher |
| run_gradattr.sh | usage: bash run_gradattr.sh <gpu> <worker_idx> <n_workers> |
| run_gradvr.sh | usage: bash run_gradvr.sh <gpu> <worker> <n_workers> |
| run_gscore.sh | run gscore.py on every captured state in gscore_20261103 (skips done); loops until CAPTURE_DONE and all processed |
| run_ledger.sh | usage: bash run_ledger.sh <gpu> <unit...>   unit = cwpair_mlp_s0 / dqn_s1 ... |
| run_pool.sh | 2026-11-04 job pool under ONE gpu_guard (GPU0-3): runs $INIT, then every 60 s starts any new $OUT/_queue/*.sh |
| run_t.sh |  |

### 停止脚本

| 文件 | 说明 |
|---|---|
| stop_all_1109.sh | stop AL 2x2 (launch_al_minatar.sh) and the G=2 seaquest backfill (launch_g2_minatar.sh); patterns only in this file |
| stop_cfpair.sh | run as a script file; never inline its pkill patterns in an ssh command line |
| stop_cwduel_A.sh | stop CW-Duel stage A (run as a script file; do not inline pkill -f in an ssh command line) |
| stop_cwduel_A2.sh | stop CW-Duel stage A (run as a script file; do not inline pkill -f in an ssh command line) |
| stop_cwduel_full.sh | run as a script file; never inline its pkill patterns in an ssh command line |
| stop_cwduel_full_lr.sh | run as a script file; never inline its pkill patterns in an ssh command line |
| stop_diag.sh |  |
| stop_dmc_smoke.sh |  |
| stop_dmc_v10.sh |  |
| stop_dmc_v9.sh | stop launcher v9 (workers + running jobs) and release our GPU holders |
| stop_duelsa.sh | 2026-11-04 stop head=duel_sa experiment (user).  guard killed with -9 so its trap does not restart the 2-GPU placeholder. |
| stop_mlpall.sh |  |
| stop_nonterm.sh | run as a script file; never inline its pkill patterns in an ssh command line |
| stop_pgx.sh | stop the main benchmark (run as: bash stop_pgx.sh — do not inline in an ssh command line) |
| stop_qcw_smc.sh | stop the softmax-centred qcw run (launch_qcw_smc.sh + its python jobs); patterns live only in this file |
| stop_qcw_smcsg.sh |  |
| stop_rc51_rho.sh |  |
| stop_sametraj.sh | run as a script file (pkill pattern must not appear in the ssh command line) |
| stop_svc.sh | run as a script file; never inline its pkill patterns in an ssh command line |
| stop_u32.sh | run as a script file (the pkill pattern must not be on the ssh command line) |
| stop_vr.sh | run as a script file; never inline its pkill patterns in an ssh command line |

### 进度查看

| 文件 | 说明 |
|---|---|
| status_cfpair.sh |  |
| status_cwduel_A.sh | progress of CW-Duel stage A: last logged timestep / eval_return / sps of every run |
| status_cwduel_A2.sh | progress of CW-Duel stage A: last logged timestep / eval_return / sps of every run |
| status_cwduel_full.sh |  |
| status_cwduel_full_lr.sh |  |
| status_dmc.sh |  |
| status_nonterm.sh |  |
| status_pgx.sh |  |
| status_sametraj.sh |  |
| status_svc.sh |  |
| status_u32.sh |  |
| status_vr.sh |  |

### 热切换（替换正在运行的启动器/占卡程序）

| 文件 | 说明 |
|---|---|
| switch_dmc_v2.sh | 11-05: replace launcher v1 by v2 without killing running training processes, placeholder stays off. |
| switch_dmc_v3.sh | 11-05 15:30: one job per GPU (segfault at build is ~100% when another job shares the GPU; diag_conc 3/3 vs idle 3/3 ok). |
| switch_mgr2.sh | 2026-11-10 switch v1 (GPU0-3, frac) -> v2 (GPU0-1, 22 GB target); free GPU2/3 completely |
| switch_mgr_012.sh | 2026-11-10 (user): hold GPU 0-2 only (release GPU3), target ~23.5 GB/card (reserve for DMControl later) |
| switch_v10.sh | 2026-11-08: move to the new holder rule.  Kill old launchers + all old holders (NOT the running python jobs), |

### 代码补丁（都已应用到 stoix/ 源码；原文件备份在 archive/code_bak/）

| 文件 | 说明 |
|---|---|
| patch_batchreset.py | 2026-11-05: cheetah slowness fix (no version change).  Playground CheetahRun.reset runs 200 'stabilise' physics steps. |
| patch_ckpt.py | 2026-11-08: optional parameter checkpoints at 20/40/60/80/100% of training (for same-state action-ranking tests). |
| patch_cmin_cont.py | 2026-11-08: level-free (centered) clipped-double min for head=qcw in cw_continuous.py (system.min_mode=raw/centered). |
| patch_cw_closed.py | 2026-11-09 patch: closed-form CW-rho with separable E1 / E2 (system.cw_closed = none/A/B/C/D), all-pairs qcw/duel path. |
| patch_cw_kernel.py | 2026-11-09 patch: gauge-neutral E1/E2 factorial via symmetric pair kernels (system.cw_kernel = none/B2/C2). |
| patch_dqn_st.py |  |
| patch_duelsa.py | 2026-11-04: add head=duel_sa to ff_rpair.py (CW-Duel with a pairwise (s,a) advantage head). |
| patch_handshake.py | 2026-11-06: GPU hand-over handshake for cw_continuous.run_experiment (env GPU_HANDSHAKE_DIR).  The launcher keeps a |
| patch_keepalive_clearcache.py | 2026-11-08: keepalive also disables jax.clear_caches (segfault in cuModuleUnload via clear_all_weakref_lru_caches, |
| patch_kinetix_cont.py | 2026-11-08: cw_continuous.py support for pytree (Kinetix EntityObservation) observations + transformer critic. |
| patch_nan_cont.py | 2026-11-05: sanitise MJX physics blow-ups in cw_continuous.env_step.  Playground DMC envs terminate (done=1) when |
| patch_nonorm_default.py | 2026-11-10 (user): nonorm becomes the default for head=qcw (explicit +system.cw_no_norm=False restores the old 1/(1-gamma) version).""" |
| patch_qcw.py | 2026-11-04: add head=qcw to ff_rpair.py: cross-weighted pair loss on a plain DQN head (no scalar c). |
| patch_qcw_al.py | 2026-11-09 (user): advantage learning (Bellemare et al. 2016, AL, beta=al_alpha) for the all-pairs heads (qcw / duel). |
| patch_qcw_center.py | 2026-11-09 (user): qcw head output Qt(s,a) = Q(s,a) - sum_b softmax(Q(s,.))_b Q(s,b)  (softmax-weighted zero mean per state, |
| patch_qcw_center_sg.py | 2026-11-09 (user): qcw_center="softmax_sg": Q - sum_b sg(softmax(Q))_b Q(s,b)  (weights without gradient) |
| patch_qcw_cont.py | 2026-11-04: continuous CW-Q-rho (head=qcw) + community-alignment options for cw_continuous.py. |
| patch_qcw_crn.py | 2026-11-09 patch: same-state different-action pairs (CRN counterfactual branch) for CW-Q-rho (head=qcw, all pairs, rho kernel). |
| patch_qcw_crn_nomix.py | 2026-11-09 patch: system.cf_same_mixed (default True).  False -> same-state twin pairs whose branches differ in |
| patch_qcw_fulltgt.py | 2026-11-10 (user): qcw all-pairs -- build the whole stop-grad target sg(D + xi/(1-g)) from the TARGET net |
| patch_qcw_nonorm.py | 2026-11-10 (user): qcw/duel all-pairs -- drop the 1/(1-gamma) normalisation of xi (delta = xi instead of xi/(1-gamma)). |
| patch_qcw_nstep.py | 2026-11-09 (user): n-step CW-Q-rho (all-pairs qcw/duel heads, same_traj + pair_rho path). |
| patch_rc51_rho.py | 2026-11-09 (user): R-C51 with the rho pairing (same_traj groups G=8, rho_pairs: all within-group ordered pairs weighted |
| patch_rho_sparse.py | 2026-11-09 (user): G=8, cross-trajectory rho weight shared by only K = G-1 partners per item (not the whole batch). |
| patch_rho_td.py | 2026-11-09 patch: terminal-decoupled rho (system.rho_term_decouple=True) for the all-pairs heads (qcw/duel). |
| patch_sac_cont.py | 2026-11-06: SAC base for cw_continuous.py (baseline only: head=q), aligned to the MuJoCo Playground Brax SAC config. |
| patch_switch.py | 2026-11-04 breakout diagnosis: loss switch on the CW-Duel (head=duel) path. |
| patch_tshuf.py | 2026-11-10 E1 causal control (user): for every TERMINAL item, its same-group (same-trajectory) pair weights are moved |

### 正确性检查（当前仍有效的已复制到 cwq/checks/）

| 文件 | 说明 |
|---|---|
| check_alpal.py | check_alpal (2026-11-03).  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_alpal.py""" |
| check_batchreset.py | 2026-11-05: BatchCondAutoResetVmap == AutoResetWrapper+VmapWrapper (CPU, CartpoleBalance + CheetahRun, 1100 steps, |
| check_c51.py | C51 projection checks (float32 tolerances: support step 0.8, errors ~2e-5).""" |
| check_cmin_cont.py | 2026-11-08 check for patch_cmin_cont.py (head=qcw, system.min_mode=centered). |
| check_cw_closed.py | 2026-11-09 checks for cw_closed: arm A gradient == existing duel_all_pairs gradient (cur_target on and off); |
| check_cw_continuous.py | check_cw_continuous (2026-11-04).  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_cw_continuous.py""" |
| check_cw_kernel.py | 2026-11-09 checks for the gauge-neutral kernels (balance_kernel; arms A / B2 / C2 / D=rho 1).""" |
| check_duel.py | Checks for head=duel (CW-Duel) in ff_rpair.py, 2026-11-02.  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_duel.py |
| check_duel_mlp.py | Checks for head=duel_mlp (CW-Duel, MLP state-pair head, K partners), 2026-11-02. |
| check_duel_sa.py | check_duel_sa (2026-11-04).  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_duel_sa.py |
| check_nonterm.py | Checks for system.nonterm_only (2026-11-02).  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_nonterm.py |
| check_nstep.py | Unit checks for nstep_loss in ff_rdqn.py (2026-10-31).""" |
| check_qcw.py | check_qcw (2026-11-04).  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_qcw.py |
| check_qcw_al.py | 2026-11-09 checks for AL in the all-pairs qcw head (system.al_mode=al).""" |
| check_qcw_center.py |  |
| check_qcw_cont.py | check_qcw_cont (2026-11-04): continuous CW-Q (head=qcw) in cw_continuous.py. |
| check_qcw_crn.py | 2026-11-09 checks for qcw same-state CRN pairs (cf_same).""" |
| check_qcw_crn_nomix.py | 2026-11-09 checks for qcw same-state CRN pairs (cf_same).""" |
| check_qcw_fulltgt.py | 2026-11-10 checks for qcw_full_target.""" |
| check_qcw_nonorm.py | 2026-11-10 checks for cw_no_norm.""" |
| check_qcw_nstep.py | 2026-11-09 checks for n-step CW-Q-rho (same_traj.nstep_tail + duel_all_pairs g_item).""" |
| check_rc51_rho.py |  |
| check_reset.py | check reset_continuing step: done stored 0; at real terminations next_obs == next transition's obs (reset obs) |
| check_rho.py | check_rho (2026-11-03): rho-kernel estimators.  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_rho.py""" |
| check_rho_duel.py | check_rho_duel (2026-11-04).  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_rho_duel.py""" |
| check_rho_sparse.py |  |
| check_rho_td.py | 2026-11-09 unit checks for same_traj.rho_weight_matrix_td.""" |
| check_rpair.py | Checks for ff_rpair.py (2026-10-31).  JAX_PLATFORMS=cpu python scripts_rdqn/check_rpair.py |
| check_rpair_cf.py | (float32: value comparisons use relative tolerance 1e-6, gradients absolute 1e-5) |
| check_same_traj.py | Checks for same_traj.same_traj_batch (2026-11-03).  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_same_traj.py""" |
| check_selfpair.py | 2026-11-09: what do self-pairs (pairs of the same buffer item, incl. the diagonal) do in the qcw CW-rho loss? |
| check_switch.py | check_switch (2026-11-04).  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_switch.py""" |
| check_tshuf.py |  |
| check_vr.py | Checks for strat_term / term_partners (2026-11-02).  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_vr.py |

### 冒烟测试

| 文件 | 说明 |
|---|---|
| smoke |  |
| smoke.log |  |
| smoke.sh |  |
| smoke2 |  |
| smoke2.sh |  |
| smoke3.sh |  |
| smokeA |  |
| smokeA.sh |  |
| smoke_base2.sh |  |
| smoke_cfpair.sh | 0.2M-step smoke of both arms on breakout + asterix (GPU $1), then prints the last records. |
| smoke_cheetah_fast.sh | 2026-11-05: cheetah with batch-cond auto-reset + no update-batch vmap: throughput (td3, cwq_rho), 3 evals of 128k... |
| smoke_cw_continuous.sh | CPU smoke (2026-11-04): CartpoleBalance (no termination, truncation at 1000) for ddpg/td3 x q/pair/duel (+ random pairing) |
| smoke_diag.sh |  |
| smoke_dm.sh | smoke of head=duel_mlp K=64 (2026-11-02): asterix + seaquest 0.2M, 4 evals, GPU ${GPU:-2} |
| smoke_dmc.sh | GPU smoke 2026-11-04: throughput + sanity for td3 / cwq_rho, 1e5 steps.  GPU0: 4 concurrent, GPU1: 1 alone. |
| smoke_duel.sh | smoke test of head=duel (CW-Duel) 2026-11-02: 0.2M steps asterix, rank 32 and 0, GPU ${GPU:-2}; records sps / memory |
| smoke_kinetix.sh | 2026-11-08 Kinetix smoke + throughput: each arm 2M steps (2 evals) on GPU $1, sequential. |
| smoke_new.sh |  |
| smoke_nonterm.sh |  |
| smoke_nstep.sh |  |
| smoke_pg.sh | 2026-11-04 smoke of the Playground-SAC-aligned protocol (dmc_pg_common.sh), 3e5 steps, logs appended, faulthandler on. |
| smoke_pgx.sh |  |
| smoke_rpair.sh |  |
| smoke_sac.sh | 2026-11-06: (1) CPU smoke ddpg/sac CartpoleBalance 3.2e4 steps; (2) GPU walker_walk ddpg/sac 6.4e5 steps (learning check) |
| smoke_sametraj.sh |  |
| smoke_sq.sh | seaquest divergence diagnosis 2026-11-02: 0.1M steps, eval every 10k; duel r0 / duel r0 cur_target / cwpair_lr |
| smoke_vr.sh |  |

### 其他（诊断、分析、GPU/驱动排查、日志）

| 文件 | 说明 |
|---|---|
| backfill_g16_mujoco.sh | 2026-11-08 backfill for launch_g16_mujoco.sh: runs that are neither done nor running get (re)started, one at a time |
| capture2_ff_rdqn.py | DQN / RDQN with an optional CRN counterfactual branch (Anakin, single file).  2026-10-30. |
| capture2_ff_rpair.py | Pairwise mean critic for MinAtar (Anakin, single file).  2026-10-31. |
| capture2_runs.sh | 2026-11-03 g-score checkpoints: {dqn, rdqn, rdqn_st, cwpair_mlp, cwpair_mlp_st} x {asterix, seaquest} x seeds 0-1, 2M real steps, |
| capture_ff_rdqn.py | DQN / RDQN with an optional CRN counterfactual branch (Anakin, single file).  2026-10-30. |
| capture_ff_rpair.py | Pairwise mean critic for MinAtar (Anakin, single file).  2026-10-31. |
| capture_runs.sh | asterix, seeds 0-1, CW-Pair MLP K=8 and DQN, 2M real steps, full learner state saved at the end (same protocol as minatar_pgx) |
| check_alpal_wiring.sh | alpha=0 with al_mode=al must reproduce al_mode=none exactly (CPU, deterministic); alpha=0.9 must log al_pen>0. |
| check_gradattr.sh |  |
| d3_readout.py | D3 (2026-11-03): read-out swap on CW-Pair MLP K=8 checkpoints, no retraining. |
| decision_ledger.py | Decision ledger (2026-11-03): what do 1000 real training updates do to the GREEDY decisions, and which training |
| det_test.sh |  |
| diag_flags.sh | 2026-11-05: which XLA flag removes the GPU compile segfault of scan(MJX step)?  HopperHop, 3 repeats per variant. |
| diag_nstep.sh |  |
| diag_segv.py | 2026-11-05 minimal segfault repro: Playground env jit reset/step (vmap 128) and a 64-step scan, no Stoix.""" |
| diag_segv.sh |  |
| dmc_cmin_switch.sh | 2026-11-08 (user): DMC CW-Q-rho(TD3) + level-free min, 10 seeds x 6 tasks, GPU 2-3 (v9 launcher, own holders stay after). |
| dmc_common.sh | 2026-11-04 DMC (MuJoCo Playground) community-aligned TD3 protocol (Fujimoto 2018 / CleanRL td3_continuous_action): |
| dmc_cwqddpg_start.sh | 2026-11-08 (user): DMC CW-Q-rho(DDPG) (arm cwq_ddpg), 6 tasks x 10 seeds. |
| dmc_pg_common.sh | 2026-11-04 DMC main protocol = MuJoCo Playground official Brax SAC config for DM Control Suite |
| fh.log |  |
| fh.sh |  |
| gap_analysis.py | DIAG 2026-11-01: action-gap / greedy-agreement analysis on parameter checkpoints (DQN vs CW-Pair). |
| gap_analysis2.py | DIAG 2026-11-01 (v2): action gaps and greedy agreement on checkpoints, DQN vs CW-Pair, all seeds at once. |
| gap_loop.sh | DIAG 2026-11-01: run gap_analysis.py (--mc) on every checkpoint of every (game, seed) whose 3 algos are done. |
| gap_loop2.sh | DIAG 2026-11-01 v2: gap_analysis2.py --mc for every checkpoint step t once all 9 agents (3 algos x seeds 0-2) of a game are done. |
| gpu_busy.sh | 2026-11-08 (v2 2026-11-10)  source in launchers.  gpu_busy_on <gpu>: mark busy (flag owned by this shell's $BASHPID) and |
| gpu_guard.sh | Run an experiment launcher on GPU0/1 with the placeholder released, and put the placeholder back afterwards |
| gpu_hold.py | 2026-11-06: GPU holder between our jobs.  Run with CUDA_VISIBLE_DEVICES=<g> XLA_PYTHON_CLIENT_PREALLOCATE=true |
| gpu_hold_mgr.sh | 2026-11-08 GPU holder manager (user rule): while OUR experiments run on a GPU, no holder there; when none of our |
| gpu_hold_mgr2.sh | 2026-11-10 GPU holder manager v2 (user): we only use GPU 0 and 1 (2-7 left to others).  When none of our experiments |
| gpu_hold_stop.sh |  |
| grad_attr.py | DIAG 2026-11-02: is the variance of terminal pairs harmful?  Gradient attribution by pair type on checkpoints. |
| grad_vr.py | DIAG 2026-11-02: offline gradient quality of the variance-reduction estimators on CW-Pair MLP K=8 checkpoints. |
| gscore.py | g-score probe (2026-11-03), adapted from Zhou et al. 2022 (camall3n/actgen change_q.py / gscore.py). |
| kinetix_pg_common.sh | 2026-11-08 Kinetix off-policy protocol (user: 1e8 env steps, transformer for all methods; PPO later). |
| mk_sac_cfg.sh | 2026-11-06: SAC entry configs (copied from the CW-TD3 ones) + DMC protocol helpers for ddpg / sac arms |
| placeholder.sh | GPU0-3 placeholder (2026-11-04: was 0,1) (user's /home/junweiluo/trainer.py).  usage: bash scripts_rdqn/placeholder.sh start/stop/status |
| prof_dmc.sh | 2026-11-04 throughput profiling: one job per GPU, persistent compilation cache. |
| ranking/ |  |
| rebalance.sh | 2026-11-08: DMC (cwq_cmin+cwq_ddpg, v10) only on GPU 0,1; Kinetix on GPU 2,3 (2 each) + GPU 0,1 once DMC frees memory. |
| rerun1.sh |  |
| rerunA.sh |  |
| restart_kinetix.sh |  |
| segv_test.sh | 2026-11-04: does the thread-stack size matter for the cheetah compile segfault?  (glibc: RLIMIT_STACK=unlimited -> |
| status_sametraj.py |  |
| stk.out |  |
| stk.sh |  |
| stk_8192.log |  |
| stk_unlimited.log |  |
| test_crn_step.log |  |
| test_crn_step.py | CRN mechanism check in Stoix/stoa gymnax (Asterix-MinAtar, full stoix wrapper stack). |
| timing_dmc.sh | 2026-11-05 per-task throughput on the main protocol (td3 arm), 2 jobs per GPU, 1.92e5 steps (3 evals of 64k). |
| xm.out |  |
| xm.sh |  |
| xm_autotune0.log |  |
| xm_both_off.log |  |
| xm_cmdbuf_off.log |  |

## scripts_mujoco/

### 启动器（每个对应一次实验；OUT=结果目录，位于 /home/junweiluo/stoix_runs/）

| 文件 | 结果目录 | 说明 |
|---|---|---|
| launch_ablation.sh | mujoco_gym | 2026-11-07 ablations (user approved): Hopper-v5, Walker2d-v5 x seeds 0-4 x 1M, CleanRL protocol. |
| launch_cmin3.sh | mujoco_gym | 2026-11-08 (user): CW-Q-rho(TD3) + level-free min on Ant/Humanoid/HalfCheetah x seeds 0-4 (15 runs) on GPU 0-1. |
| launch_cwq_ddpg.sh | mujoco_gym | 2026-11-07 extra arm (user): CW-Q-rho on a DDPG base (one critic, no smoothing, CleanRL DDPG otherwise) = |
| launch_mujoco.sh | mujoco_gym | 2026-11-07 v3 (GPU)  Gym MuJoCo v5, CleanRL-default protocol: 4 arms x 5 envs x 5 seeds x 1M steps. |
| launch_mujoco_nomin.sh | mujoco_gym | 2026-11-08 (user): CW-Q-rho(TD3) WITHOUT clipped double Q ("raw": each critic bootstraps from its own target), |

### 停止脚本

| 文件 | 说明 |
|---|---|
| stop_holders_mj.sh |  |
| stop_mujoco.sh |  |

### 热切换（替换正在运行的启动器/占卡程序）

| 文件 | 说明 |
|---|---|
| switch_v2.sh |  |
| switch_v3.sh |  |

### 正确性检查（当前仍有效的已复制到 cwq/checks/）

| 文件 | 说明 |
|---|---|
| check_mujoco.py | Checks for mujoco_offpolicy.py (2026-11-07).  Run: JAX_PLATFORMS=cpu python check_mujoco.py""" |

### 冒烟测试

| 文件 | 说明 |
|---|---|
| smoke |  |
| smoke.nohup |  |
| smoke.sh |  |

### 其他（诊断、分析、GPU/驱动排查、日志）

| 文件 | 说明 |
|---|---|
| gputest/ |  |
| gputest.nohup |  |
| gputest.sh |  |
| launch.nohup |  |
| launch_nomin.nohup |  |
| mujoco_offpolicy.py | Gym MuJoCo (Gymnasium v5) off-policy baselines + CW-Q-rho, single file, JAX on CPU.  2026-11-07. |

## scripts_rppo/

### 启动器（每个对应一次实验；OUT=结果目录，位于 /home/junweiluo/stoix_runs/）

| 文件 | 结果目录 | 说明 |
|---|---|---|
| launch_rgae.sh | minatar_rppo_10m | R-PPO + R-GAE arm (2026-11-01): same protocol as launch_rppo.sh, actor advantages = RVL R-GAE (zeros init). |
| launch_rppo.sh | minatar_rppo_10m | PPO vs R-PPO (2026-11-01): {asterix, breakout} x {ppo, rppo} x seeds 0-4 x 10M env steps (gymnax MinAtar). |

### 停止脚本

| 文件 | 说明 |
|---|---|
| stop_rppo.sh |  |

### 进度查看

| 文件 | 说明 |
|---|---|
| status_rppo.sh |  |

### 冒烟测试

| 文件 | 说明 |
|---|---|
| smoke_rgae.sh |  |

### 其他（诊断、分析、GPU/驱动排查、日志）

| 文件 | 说明 |
|---|---|
| check_rppo.sh | Equivalence check (2026-11-01): ff_rppo with rppo=False must reproduce stock ff_ppo (same seed) on asterix; |
| check_rppo_cpu.sh |  |
| kill_rgae.sh |  |


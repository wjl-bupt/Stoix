"""Generate archive/README.md (index of the archived scripts)."""
import os, re, glob
A = "/home/junweiluo/Stoix/archive"
def first_comment(f):
    try:
        L = open(f, errors="replace").read().splitlines()
    except Exception:
        return ""
    for ln in L[:12]:
        s = ln.strip()
        if s.startswith("#!") or not s:
            continue
        if s.startswith("#") or s.startswith('"""') or s.startswith("'''"):
            return re.sub(r'^[#"\' ]+', "", s)[:160]
        break
    return ""
def outdir(f):
    m = re.search(r"OUT=\$\{OUT:-([^}]+)\}|OUT=(/home/junweiluo/stoix_runs/[^ ;]+)", open(f, errors="replace").read())
    return (m.group(1) or m.group(2)).replace("/home/junweiluo/stoix_runs/", "") if m else ""
cats = [("launch_*.sh", "启动器（每个对应一次实验；OUT=结果目录，位于 /home/junweiluo/stoix_runs/）"),
        ("start_*.sh", "启动器包装"), ("run_*.sh", "批处理/分析驱动"), ("stop_*.sh", "停止脚本"), ("status_*.sh", "进度查看"),
        ("switch_*.sh", "热切换（替换正在运行的启动器/占卡程序）"), ("patch_*.py", "代码补丁（都已应用到 stoix/ 源码；原文件备份在 archive/code_bak/）"),
        ("check_*.py", "正确性检查（当前仍有效的已复制到 cwq/checks/）"), ("smoke*", "冒烟测试")]
out = ["# archive/ — 2026-11-10 之前的脚本归档", "",
       "2026-11-10 整理仓库时，`scripts_rdqn/`、`scripts_mujoco/`、`scripts_rppo/` 原样移到这里（文件内容未改，脚本里的路径",
       "仍指向旧位置，所以**不要直接运行**，需要复现旧实验时先改路径或用 `cwq/` 的启动器）。新的入口见 `cwq/README.md`。",
       "", "| 目录 | 内容 |", "|---|---|",
       "| scripts_rdqn/ | MinAtar / DMC / Kinetix 的全部旧启动器、补丁、检查、诊断与分析脚本（10-30 至 11-10） |",
       "| scripts_rdqn/ranking/ | 同状态排序评测、Breakout 漏球、κ 噪声、E1/E2、估计量诊断、线性探针等分析脚本 |",
       "| scripts_mujoco/ | Gym MuJoCo 旧启动器（实现已移到 cwq/mujoco_gym/mujoco_offpolicy.py） |",
       "| scripts_rppo/ | relative PPO（rppo）实验脚本 |",
       "| code_bak/ | stoix/ 与 scripts 下所有 *.bak_* 源码备份，按原相对路径存放 |",
       "| patches_20261110/ | 本次整理时应用的补丁（连续动作 nonorm） |", ""]
for d in ["scripts_rdqn", "scripts_mujoco", "scripts_rppo"]:
    out.append(f"## {d}/"); out.append("")
    files = sorted(os.listdir(f"{A}/{d}"))
    used = set()
    for pat, title in cats:
        fs = [f for f in files if glob.fnmatch.fnmatch(f, pat) and not ".bak" in f]
        if not fs:
            continue
        used |= set(fs)
        out.append(f"### {title}"); out.append("")
        if pat == "launch_*.sh":
            out.append("| 文件 | 结果目录 | 说明 |"); out.append("|---|---|---|")
            for f in fs:
                out.append(f"| {f} | {outdir(f'{A}/{d}/{f}')} | {first_comment(f'{A}/{d}/{f}').replace('|', '/')} |")
        else:
            out.append("| 文件 | 说明 |"); out.append("|---|---|")
            for f in fs:
                out.append(f"| {f} | {first_comment(f'{A}/{d}/{f}').replace('|', '/')} |")
        out.append("")
    rest = [f for f in files if f not in used and not f.startswith("__") and ".bak" not in f]
    if rest:
        out.append("### 其他（诊断、分析、GPU/驱动排查、日志）"); out.append("")
        out.append("| 文件 | 说明 |"); out.append("|---|---|")
        for f in rest:
            p = f"{A}/{d}/{f}"
            out.append(f"| {f}{'/' if os.path.isdir(p) else ''} | {'' if os.path.isdir(p) else first_comment(p).replace('|', '/')} |")
        out.append("")
open(f"{A}/README.md", "w").write("\n".join(out) + "\n"); print("archive README lines:", len(out))

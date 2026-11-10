"""Learning curves (mean +- s.e. over seeds) + AUC/final table for any set of runs.   (2026-11-10)

usage:
  python cwq/analysis/plot_curves.py --units asterix breakout freeway seaquest space_invaders \
      --run "CW-Q-rho (ours)=/home/junweiluo/stoix_runs/minatar_main:cwq" \
      --run "DQN=/home/junweiluo/stoix_runs/minatar_main:dqn"  ... --out /tmp/fig
  Run spec  LABEL=DIR:ALGO  reads DIR/<unit>_<ALGO>_s<seed>.jsonl for every seed found (Stoix logs: eval_return vs
  timestep/env_calls; Gym MuJoCo script logs: eval_return vs step).  The first --run is drawn thick (ours).
Outputs: <out>/curves.png (top row env steps, bottom row simulator calls when available) and <out>/summary.txt
  (AUC = mean of all evaluations, final = mean of the last 5)."""
import argparse, glob, json, os, re
import numpy as np


def load(path):
    rows = []
    for ln in open(path):
        r = json.loads(ln)
        if "eval_return" in r and r["eval_return"] is not None:
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", required=True); ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--title", default="")
    A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
    runs = []
    for spec in A.run:
        lab, rest = spec.split("=", 1); d, algo = rest.rsplit(":", 1); runs.append((lab, d, algo))
    data = {}
    for u in A.units:
        for lab, d, algo in runs:
            fs = sorted(glob.glob(f"{d}/{u}_{algo}_s*.jsonl"), key=lambda f: int(re.search(r"_s(\d+)\.jsonl$", f).group(1)))
            fs = [f for f in fs if re.fullmatch(rf"{re.escape(u)}_{re.escape(algo)}_s\d+\.jsonl", os.path.basename(f))]
            data[(u, lab)] = [x for x in (load(f) for f in fs) if x]
    L = ["AUC (mean of all evaluations) / final (mean of last 5), mean +- s.e. over seeds, (n seeds)", ""]
    L.append(f"{'':28s}" + "".join(f"{u:26s}" for u in A.units))
    for lab, _, _ in runs:
        row = f"{lab:28s}"
        for u in A.units:
            S = np.array([[np.mean([r["eval_return"] for r in x]), np.mean([r["eval_return"] for r in x[-5:]])] for x in data[(u, lab)]])
            if len(S) == 0:
                row += f"{'-':26s}"; continue
            se = S[:, 0].std(ddof=1) / np.sqrt(len(S)) if len(S) > 1 else float("nan")
            row += f"{S[:, 0].mean():.1f}+-{se:.1f}/{S[:, 1].mean():.1f} ({len(S)})".ljust(26)
        L.append(row)
    txt = "\n".join(L); print(txt); open(f"{A.out}/summary.txt", "w").write(txt + "\n")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
    xs = [k for k in ("timestep", "env_calls") if any(k in x[0][0] for x in data.values() if x)] or ["step"]
    fig, ax = plt.subplots(len(xs), len(A.units), figsize=(5 * len(A.units), 4 * len(xs)), squeeze=False)
    for j, u in enumerate(A.units):
        for i, xk in enumerate(xs):
            for c, (lab, _, _) in enumerate(runs):
                R = data[(u, lab)]
                if not R:
                    continue
                k = min(map(len, R)); xk_ = xk if xk in R[0][0] else "step"
                x = np.array([r[xk_] for r in R[0][:k]]); Y = np.array([[q["eval_return"] for q in r[:k]] for r in R])
                m = Y.mean(0); se = Y.std(0, ddof=1) / np.sqrt(len(R)) if len(R) > 1 else 0 * m
                col = "C3" if c == 0 else f"C{[0, 1, 2, 4, 5, 6, 7, 8, 9][(c - 1) % 9]}"
                ax[i, j].plot(x, m, color=col, lw=2.4 if c == 0 else 1.2, label=f"{lab} (n={len(R)})", zorder=5 if c == 0 else 2)
                ax[i, j].fill_between(x, m - se, m + se, color=col, alpha=.18 if c == 0 else .08)
            ax[i, j].set_title(u); ax[i, j].set_xlabel({"timestep": "env steps", "env_calls": "simulator calls"}.get(xk, xk))
            ax[i, j].set_ylabel("eval return"); ax[i, j].grid(alpha=.3)
    h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=min(8, len(runs)), frameon=False)
    if A.title:
        fig.suptitle(A.title)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97]); fig.savefig(f"{A.out}/curves.png", dpi=110); print("wrote", A.out)


if __name__ == "__main__":
    main()

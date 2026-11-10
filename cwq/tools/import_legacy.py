"""Copy finished legacy runs into a launcher's results dir under the launcher's run names, after checking that the run
was made with exactly the settings the launcher would use now.   (2026-11-10)

usage:  DRY_RUN=1 [ALGOS=... other launcher env] bash cwq/run_<bench>.sh > /tmp/dry.txt
        python cwq/tools/import_legacy.py /tmp/dry.txt  <dst_algo>=<legacy_dir>:<legacy_algo> [...]  [--apply]

For every queued run of <dst_algo> in the dry-run listing, the legacy file <legacy_dir>/<unit>_<legacy_algo>_s<seed>.jsonl
is compared key by key with what the launcher would run now, in both directions:
  Stoix runs     : the full config is composed with Hydra (system file's config + the command's overrides) and every leaf
                   of system / network / env is compared with the legacy config; arch.* overrides are compared too.
  Gym MuJoCo runs: every argparse option (command flag, else the script's current default) vs the legacy vars(args).
Keys absent from an old config take the old code default (LEGACY_DEFAULT); other new keys must be at an "off" value.
Only finished, fully matching runs are copied (never overwriting); without --apply nothing is written.
A record of every import is appended to <OUT>/_imported.txt."""
import json, os, re, shutil, sys
import yaml
STOIX = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SKIP = {"env", "system.run_log_dir", "system.algo_name", "arch.seed", "system.ckpt_dir"}
SKIP_FLAGS = {"out", "seed", "env", "ckpt_dir"}
LEGACY_DEFAULT = {"system.cw_no_norm": False, "system.no_norm": False, "system.min_mode": "raw", "no_norm": 0,
                  "min_mode": "raw"}


def same(a, b):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b) if isinstance(a, bool) and isinstance(b, bool) else str(a).lower() == str(b).lower()
    try:
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
    except (TypeError, ValueError):
        return str(a) == str(b)


def lookup(cfg, key):
    x = cfg
    for k in key.split("."):
        if not isinstance(x, dict) or k not in x:
            return KeyError
        x = x[k]
    return x


OFF = (False, 0, 0.0, None, "none", "raw", "", "uniform")
# keys that only take effect when system.same_traj=True (grouped sampling); added to the defaults after 10-31
SAME_TRAJ_ONLY = {"system.st_group", "system.rho_kc"}


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        kk = f"{pre}.{k}" if pre else k
        if isinstance(v, dict):
            out.update(flat(v, kk))
        else:
            out[kk] = v
    return out


def compose_stoix(cmd):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    toks = cmd.split(); py = next(t for t in toks if t.endswith(".py"))
    src = open(os.path.join(STOIX, py)).read()
    m = re.search(r'@hydra\.main\(\s*config_path="([^"]+)",\s*config_name="([^"]+)"', src)
    cdir = os.path.normpath(os.path.join(STOIX, os.path.dirname(py), m.group(1)))
    ovr = [t for t in toks[toks.index(py) + 1:] if "=" in t]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=cdir, version_base="1.2"):
        cfg = compose(config_name=m.group(2), overrides=ovr)
    return OmegaConf.to_container(cfg, resolve=False)


def check_full_stoix(cmd, cfg):
    new = flat(compose_stoix(cmd)); old = flat(cfg); bad = []
    for k, v in new.items():
        if not k.startswith(("system.", "network.", "env.")) or k in SKIP:
            continue
        if v is None or (isinstance(v, str) and ("${" in v or v == "???")):
            continue
        if k in SAME_TRAJ_ONLY and not new.get("system.same_traj", False):
            continue
        if k not in old:
            if k in LEGACY_DEFAULT:
                if not same(LEGACY_DEFAULT[k], v):
                    bad.append(f"{k}: legacy default {LEGACY_DEFAULT[k]!r} != {v!r}")
            elif not any(same(v, o) for o in OFF if not (isinstance(v, str) ^ isinstance(o, str))):
                bad.append(f"{k}: missing in legacy, new value {v!r}")
            continue
        if not same(old[k], v):
            bad.append(f"{k}: legacy {old[k]!r} != {v!r}")
    return bad


def check_full_flags(cmd, cfg):
    toks = cmd.split(); py = next(t for t in toks if t.endswith(".py"))
    src = open(py if os.path.isabs(py) else os.path.join(STOIX, "cwq/mujoco_gym", py)).read()
    dflt = {m.group(1): yaml.safe_load(m.group(2).replace("_", "")) for m in
            re.finditer(r'add_argument\("--(\w+)"[^)]*?default=("[^"]*"|[-\w.]+)', src)}
    given = {toks[i][2:]: yaml.safe_load(toks[i + 1]) for i in range(len(toks) - 1) if toks[i].startswith("--")}
    bad = []
    for k in set(dflt) | set(cfg):
        if k in SKIP_FLAGS:
            continue
        new = given.get(k, dflt.get(k, KeyError)); old = cfg.get(k, LEGACY_DEFAULT.get(k, dflt.get(k, KeyError)))
        if new is KeyError or old is KeyError or not same(old, new):
            bad.append(f"--{k}: legacy {old!r} != {new!r}")
    return bad


def check_stoix(cmd, cfg):
    bad = []
    for w in cmd.split():
        if "=" not in w or w.startswith(("XLA_", "TF_", "CUDA_", "JAX_", "PYTHON")):
            continue
        k, v = w.lstrip("+").split("=", 1)
        if k in SKIP:
            continue
        have = lookup(cfg, k)
        if have is KeyError:
            if k in LEGACY_DEFAULT:
                have = LEGACY_DEFAULT[k]
            else:
                bad.append(f"{k}: missing (want {v})"); continue
        if isinstance(have, dict):          # config-group selection such as network=minatar_dqn
            continue
        want = yaml.safe_load(v) if v != "" else ""
        if not same(have, want):
            bad.append(f"{k}: legacy {have!r} != {want!r}")
    return bad


def check_flags(cmd, cfg):
    bad = []
    toks = cmd.split()
    for i, w in enumerate(toks):
        if not w.startswith("--"):
            continue
        k = w[2:]; v = toks[i + 1] if i + 1 < len(toks) else ""
        if k in SKIP_FLAGS:
            continue
        have = cfg.get(k, LEGACY_DEFAULT.get(k, KeyError))
        if have is KeyError:
            bad.append(f"--{k}: missing"); continue
        if not same(have, yaml.safe_load(v)):
            bad.append(f"--{k}: legacy {have!r} != {v!r}")
    return bad


def main():
    args = [a for a in sys.argv[1:] if a != "--apply"]; apply = "--apply" in sys.argv
    dry, maps = args[0], dict(a.split("=", 1) for a in args[1:])
    lines = open(dry).read().splitlines()
    out_dir = None; n_ok = n_bad = n_miss = n_have = 0
    for i, ln in enumerate(lines):
        m = re.match(r"\[(\S+) (\S+) s(\d+)\] env (.*)$", ln)
        if not m:
            continue
        unit, algo, seed, cmd = m.group(1), m.group(2), m.group(3), m.group(4)
        dst = lines[i + 1].split("-> ", 1)[1].strip(); out_dir = os.path.dirname(dst)
        name = os.path.basename(dst)[: -len(f"_s{seed}.jsonl")]
        run_algo = name.split("_", 1)[1] if not name.startswith(unit) else name[len(unit) + 1:]
        prefix = os.path.basename(dst)[: -len(f"_{run_algo}_s{seed}.jsonl")]
        if algo not in maps and run_algo not in maps:
            continue
        ldir, lalgo = maps.get(run_algo, maps.get(algo)).split(":")
        src = os.path.join(ldir, f"{prefix}_{lalgo}_s{seed}.jsonl")
        if not os.path.exists(src):
            n_miss += 1; continue
        with open(src) as fh:
            first = json.loads(fh.readline())
        last = open(src).read().rstrip().splitlines()[-1]
        if '"done"' not in last:
            n_miss += 1; print("unfinished", src); continue
        cfg = first.get("config", first)
        if " --base " in cmd:
            bad = check_flags(cmd, cfg) + check_full_flags(cmd, cfg)
        else:
            bad = check_stoix(cmd, cfg) + check_full_stoix(cmd, cfg.get("config", cfg) if "system" not in cfg else cfg)
        if bad:
            n_bad += 1; print(f"MISMATCH {src} -> {dst}:\n   " + "\n   ".join(bad)); continue
        n_ok += 1
        if os.path.exists(dst):
            n_have += 1; continue
        if apply:
            shutil.copy2(src, dst)
            with open(os.path.join(out_dir, "_imported.txt"), "a") as fh:
                fh.write(f"{src} -> {dst}\n")
    print(f"matching {n_ok} (already present {n_have}), mismatching {n_bad}, legacy missing/unfinished {n_miss}; "
          f"{'copied' if apply else 'dry run, nothing copied'}  OUT={out_dir}")


if __name__ == "__main__":
    main()

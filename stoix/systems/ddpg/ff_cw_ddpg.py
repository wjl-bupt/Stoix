"""CW-Pair / CW-Duel (and the plain-Q baseline) on a DDPG base, MuJoCo Playground DMC etc.  2026-11-04.

  python stoix/systems/ddpg/ff_cw_ddpg.py env=mjc_playground/dm_control/cheetah_run system.head=duel|pair|q \
      system.pairing=rho|random system.pair_rho=0.5 arch.seed=0 system.run_log_dir=...
All logic (networks, cross-weighted pair losses, rho pairing, truncation bootstrap, actor read-outs) is in
cw_continuous.py; see its docstring.  This file only fixes the base algorithm to DDPG."""
import keepalive  # noqa: F401  (11-06: must come before any jax user; same directory)

import sys

import hydra
from colorama import Fore, Style
from omegaconf import DictConfig, OmegaConf

from stoix.systems.ddpg.cw_continuous import run_experiment


@hydra.main(config_path="../../configs/default/anakin", config_name="default_ff_cw_ddpg.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    OmegaConf.set_struct(cfg, False)
    perf = run_experiment(cfg, base="ddpg")
    print(f"{Fore.CYAN}{Style.BRIGHT}CW-DDPG ({cfg.system.head}) experiment completed{Style.RESET_ALL}")
    keepalive.hard_exit(0)   # 11-06: skip teardown (driver crash in cuModuleUnload)
    return perf


if __name__ == "__main__":
    hydra_entry_point()

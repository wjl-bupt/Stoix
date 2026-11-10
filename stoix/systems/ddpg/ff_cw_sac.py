"""SAC baseline (head=q only) on the cw_continuous pipeline, MuJoCo Playground DMC etc.  2026-11-06.
All logic is in cw_continuous.py (base='sac', see patch_sac_cont.py)."""
import keepalive  # noqa: F401  (11-06: must come before any jax user; same directory)

import hydra
from colorama import Fore, Style
from omegaconf import DictConfig, OmegaConf

from stoix.systems.ddpg.cw_continuous import run_experiment


@hydra.main(config_path="../../configs/default/anakin", config_name="default_ff_cw_sac.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    OmegaConf.set_struct(cfg, False)
    perf = run_experiment(cfg, base="sac")
    print(f"{Fore.CYAN}{Style.BRIGHT}CW-SAC ({cfg.system.head}) experiment completed{Style.RESET_ALL}")
    keepalive.hard_exit(0)   # 11-06: skip teardown (driver crash in cuModuleUnload)
    return perf


if __name__ == "__main__":
    hydra_entry_point()

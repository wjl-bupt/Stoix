#!/bin/bash
# 2026-11-06: SAC entry configs (copied from the CW-TD3 ones) + DMC protocol helpers for ddpg / sac arms
cd /home/junweiluo/Stoix
cp /home/junweiluo/diag9/s4/ff_cw_sac.py stoix/systems/ddpg/ff_cw_sac.py
sed -e 's/^system_name: ff_cw_td3/system_name: ff_cw_sac/' -e 's/^head: duel /head: q    /' stoix/configs/system/ddpg/ff_cw_td3.yaml > stoix/configs/system/ddpg/ff_cw_sac.yaml
printf "# --- SAC (cw_continuous base=sac, 2026-11-06; Brax/Playground defaults) ---\ninit_alpha: 1.0\ntarget_entropy_scale: 0.5   # target entropy = -scale * |A|\n" >> stoix/configs/system/ddpg/ff_cw_sac.yaml
sed -e 's#system: ddpg/ff_cw_td3#system: ddpg/ff_cw_sac#' stoix/configs/default/anakin/default_ff_cw_td3.yaml > stoix/configs/default/anakin/default_ff_cw_sac.yaml
cp scripts_rdqn/dmc_pg_common.sh scripts_rdqn/dmc_pg_common.sh.bak_20261106
.venv/bin/python - <<'P'
p="scripts_rdqn/dmc_pg_common.sh"; s=open(p).read()
old="""    cwq_rho)  echo "stoix/systems/ddpg/ff_cw_td3.py $DMC_C system.head=qcw system.pairing=rho system.pair_rho=0.5 system.st_group=8" ;;"""
new=old+"""
    ddpg)     echo "stoix/systems/ddpg/ff_cw_ddpg.py ${DMC_C/system.policy_noise=0.2 system.noise_clip=0.5 system.policy_frequency=2 /} system.head=q" ;;
    sac)      echo "stoix/systems/ddpg/ff_cw_sac.py $DMC_C system.head=q arch.evaluation_greedy=True" ;;"""
assert s.count(old)==1; s=s.replace(old,new)
s=s.replace("#   Deviations from Playground:", "#   11-06: arms ddpg (one critic, no smoothing, actor every step) and sac (Brax-style, auto alpha) added.\n#   Deviations from Playground:")
open(p,"w").write(s); print("ok")
P
grep -n "ddpg)\|sac)" scripts_rdqn/dmc_pg_common.sh
source scripts_rdqn/dmc_pg_common.sh; dmc_cmd ddpg | grep -o "policy_noise" || echo "ddpg cmd has no td3 keys"; dmc_cmd sac | cut -c1-60

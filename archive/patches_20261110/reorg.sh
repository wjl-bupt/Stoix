#!/bin/bash
# 2026-11-10 repository reorganisation (user request).  Idempotent-ish: refuses if archive/ already exists.
set -e
S=/home/junweiluo/Stoix; cd $S
[ -d archive ] && { echo "archive/ exists, abort"; exit 1; }
mkdir -p archive/code_bak archive/patches_20261110
# 1. source backups (*.bak_*) out of stoix/ and the scripts dirs, keeping relative paths
find stoix scripts_rdqn scripts_mujoco scripts_rppo -name "*.bak_*" -o -name "*.bak" | grep -v "/__pycache__/" > /tmp/bak_list.txt || true
while read f; do mkdir -p archive/code_bak/$(dirname $f); mv $f archive/code_bak/$f; done < /tmp/bak_list.txt
echo "moved $(wc -l < /tmp/bak_list.txt) backup files"
# 2. new cwq/ tree (uploaded to /tmp/cwq_new)
cp -r /tmp/cwq_new $S/cwq
cp scripts_mujoco/mujoco_offpolicy.py cwq/mujoco_gym/mujoco_offpolicy.py
cp scripts_mujoco/check_mujoco.py cwq/checks/
for f in check_same_traj check_rho check_rho_sparse check_rpair check_c51 check_qcw check_qcw_nstep check_qcw_nonorm check_qcw_cont check_cmin_cont; do
  [ -f scripts_rdqn/$f.py ] && cp scripts_rdqn/$f.py cwq/checks/ || echo "missing $f"; done
sed -i 's#scripts_rdqn/check_#cwq/checks/check_#g; s#scripts_mujoco/check_#cwq/checks/check_#g' cwq/checks/*.py
sed -i 's#/home/junweiluo/Stoix/scripts_mujoco/mujoco_offpolicy.py#/home/junweiluo/Stoix/cwq/mujoco_gym/mujoco_offpolicy.py#g; s#scripts_mujoco/mujoco_offpolicy#cwq/mujoco_gym/mujoco_offpolicy#g' cwq/checks/check_mujoco.py
# 3. archive the old script dirs
mv scripts_rdqn scripts_mujoco scripts_rppo archive/
cp /tmp/cwq_patch/*.py /tmp/cwq_patch/reorg.sh archive/patches_20261110/
chmod +x cwq/*.sh cwq/tools/*.sh
echo reorg done

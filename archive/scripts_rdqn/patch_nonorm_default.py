"""2026-11-10 (user): nonorm becomes the default for head=qcw (explicit +system.cw_no_norm=False restores the old 1/(1-gamma) version)."""
import shutil
F = "/home/junweiluo/Stoix/stoix/systems/q_learning/ff_rpair.py"
shutil.copy(F, F + ".bak_20261110_before_nonorm_default")
s = open(F).read()
a = 'no_norm=bool(config.system.get("cw_no_norm", False)),'
assert s.count(a) == 2
s = s.replace(a, 'no_norm=bool(config.system.get("cw_no_norm", str(config.system.head) == "qcw")),   # 2026-11-10 default ON for qcw')
open(F, "w").write(s); print("default patched")

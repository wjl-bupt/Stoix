"""2026-11-09 patch: system.cf_same_mixed (default True).  False -> same-state twin pairs whose branches differ in
termination (d^a != d^b; delta carries the 1/(1-g) factor) get weight 0 (diagonal mask -> kernel stays symmetric,
gauge neutral).  Diagnostic control for the CW-Q-rho + CRN failure.  Backup ff_rpair.py.bak_20261109_before_nomix."""
import shutil
FR = "/home/junweiluo/Stoix/stoix/systems/q_learning/ff_rpair.py"
s = open(FR).read()
if "cf_mixed" in s:
    print("already patched"); raise SystemExit
shutil.copy(FR, FR + ".bak_20261109_before_nomix")
old = "                   cf_same: bool = False):"
assert s.count(old) == 1
s = s.replace(old, "                   cf_same: bool = False, cf_mixed: bool = True):")
old = "        cI = jnp.eye(m, dtype=pm.dtype) * (same_coef / (2.0 * m))\n"
assert s.count(old) == 1
s = s.replace(old, old + "        if not cf_mixed:                                                # 2026-11-09: drop twins that differ in termination\n"
                         "            cI = cI * (d_real == d_cf).astype(pm.dtype)[:, None]\n")
old = """                                   cf_same=bool(config.system.crn) and str(config.system.head) == "qcw",
"""
assert s.count(old) == 1
s = s.replace(old, old + """                                   cf_mixed=bool(config.system.get("cf_same_mixed", True)),
""")
open(FR, "w").write(s)
print("patched")

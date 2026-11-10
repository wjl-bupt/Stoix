"""2026-11-08: keepalive also disables jax.clear_caches (segfault in cuModuleUnload via clear_all_weakref_lru_caches,
seen on Kinetix runs sharing a GPU).  Appends to stoix/systems/ddpg/keepalive.py (backup .bak_20261108)."""
import shutil
P = "stoix/systems/ddpg/keepalive.py"
s = open(P).read()
if "clear_caches" not in s:
    shutil.copy(P, P + ".bak_20261108")
    s += '''

# 2026-11-08: never clear JAX caches (clear_caches -> executable destruction -> driver heap corruption / segfault)
import jax._src.api as _jax_api
_noop = lambda *a, **k: None  # noqa: E731
_jax_api.clear_caches = _noop
jax.clear_caches = _noop
'''
    open(P, "w").write(s)
print("ok")

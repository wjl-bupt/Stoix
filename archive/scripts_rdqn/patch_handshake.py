"""2026-11-06: GPU hand-over handshake for cw_continuous.run_experiment (env GPU_HANDSHAKE_DIR).  The launcher keeps a
holder process on the GPU between jobs; the job does its slow imports/config first, writes <dir>/ready, waits for
<dir>/go (launcher kills the holder, then touches go), initialises the GPU backend at once (with
XLA_PYTHON_CLIENT_PREALLOCATE=true it reserves its memory fraction immediately) and writes <dir>/allocated.
=> the GPU is unreserved only for ~1-3 s per job (11-06: other users grabbed GPU0/1/3 while they were free).
No env var -> unchanged behaviour."""
import os, shutil
P = "stoix/systems/ddpg/cw_continuous.py"; BAK = P + ".bak_20261106_before_handshake"
if not os.path.exists(BAK): shutil.copy(P, BAK)
s = open(BAK).read()
old = """def run_experiment(_config: DictConfig, base: str) -> float:
    config = copy.deepcopy(_config)
"""
new = """def _gpu_handshake():
    hs = os.environ.get("GPU_HANDSHAKE_DIR")
    if not hs:
        return
    open(os.path.join(hs, "ready"), "w").close()
    while not os.path.exists(os.path.join(hs, "go")):
        time.sleep(0.2)
    jnp.zeros(1).block_until_ready()          # backend init -> preallocation happens here
    open(os.path.join(hs, "allocated"), "w").close()


def run_experiment(_config: DictConfig, base: str) -> float:
    _gpu_handshake()
    config = copy.deepcopy(_config)
"""
assert s.count(old) == 1; s = s.replace(old, new); open(P, "w").write(s); print("patched", P)

"""2026-11-06: GPU holder between our jobs.  Run with CUDA_VISIBLE_DEVICES=<g> XLA_PYTHON_CLIENT_PREALLOCATE=true
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90; reserves the memory, writes argv[1] when held, then sleeps forever."""
import sys, time, jax.numpy as jnp
jnp.zeros(1).block_until_ready()
open(sys.argv[1], "w").close()
while True:
    time.sleep(3600)

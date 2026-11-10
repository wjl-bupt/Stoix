#!/bin/bash
for p in $(pgrep -f "smoke_dmc.sh|prof_dmc.sh|smoke_pg.sh|segv_test.sh|timing_dmc.sh|diag_flags.sh|diag_segv.sh|launch_dmc.sh|run_dmc_pool.sh"); do kill $p; done
sleep 1
for p in $(pgrep -f "ff_cw_td3.py"); do kill -9 $p; done

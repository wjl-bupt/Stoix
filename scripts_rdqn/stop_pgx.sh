# stop the main benchmark (run as: bash stop_pgx.sh — do not inline in an ssh command line)
pkill -f launch_minatar_pgx.sh; sleep 1; pkill -f "run_log_dir=/home/junweiluo/stoix_runs/minatar_pgx"
date > /home/junweiluo/stoix_runs/minatar_pgx/STOPPED

OUT=/home/junweiluo/stoix_runs/minatar_pgx
echo "done: $(grep -c DONE $OUT/_logs/progress.txt 2>/dev/null)/450  fail-attempts: $(grep -c FAIL $OUT/_logs/progress.txt 2>/dev/null)"
for a in dqn ddqn dueling rc c51 pqn cwpair_mlp cwpair_lr rc51; do echo "$a $(grep DONE $OUT/_logs/progress.txt 2>/dev/null | grep -c " $a s")"; done
ls $OUT/ALLDONE 2>/dev/null

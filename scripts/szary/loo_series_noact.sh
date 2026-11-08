#!/bin/bash
# LOO series for the V3-256 transition-nonlinearity ablation (configs/README_f256_noact.md):
# for each pre-registered epoch, wait until the numbered checkpoint exists and is stable, submit the
# standalone 1-GPU LOO job (scripts/szary/loo_hn_f256_noact.sh), wait for it, aggregate.
# Runs on bury under nohup from the repo root:
#   export VENV=/storage_ssd_1/szlukasik/venvs/mim-f256
#   nohup scripts/szary/loo_series_noact.sh > logs/loo_series_noact.log 2>&1 &
# Epochs (0-indexed checkpoint names): e9 collapse check, then e29, e49, e79; extend EPOCHS if the run
# continues past e79.
cd /home/szlukasik/immu-vis/multiplex-image-model/mim-f256
: "${VENV:?set VENV to the szary venv directory}"
EPOCHS=${EPOCHS:-"9 29 49 79"}
RUN=equiv-v3-int64-zs-hn-f256-noact
CKDIR=checkpoints_equiv_zs_hn_f256_noact
SERIES=logs/loo_f256_noact_series.txt
stamp() { date '+%F %T'; }
# the checkpoint dir is on /storage_ssd_1, visible from szary only. Run small commands as a step of a running
# job of mine (the training job): a new allocation would wait forever while the QOS's 2 running jobs are taken.
onszary() { local j=$(squeue -u szlukasik -t R -h -o "%i" | head -1)
            if [ -n "$j" ]; then srun --jobid=$j --overlap -n1 --cpus-per-task=1 "$@"
            else srun -n1 --partition=common --qos=szlukasik --cpus-per-task=1 --mem=8G --time=00:15:00 "$@"; fi; }
ck_size() { onszary stat -c %s "$CKDIR/checkpoint-$RUN-epoch_$1.pth" 2>/dev/null || echo 0; }
for E in $EPOCHS; do
    [ -f validation_results/loo_hn_f256_noact_e$E/checkpoint-$RUN-epoch_${E}_full.csv ] && continue
    s0=$(ck_size $E)
    while [ "$s0" = 0 ]; do sleep 900; s0=$(ck_size $E); done
    sleep 120; s1=$(ck_size $E)
    while [ "$s1" != "$s0" ]; do s0=$s1; sleep 120; s1=$(ck_size $E); done
    j=$(EPOCH=$E sbatch --parsable scripts/szary/loo_hn_f256_noact.sh)
    echo "[$(stamp)] e$E checkpoint stable ($s1 bytes); LOO submitted as job $j" | tee -a $SERIES
    while squeue -h -j "$j" 2>/dev/null | grep -q "$j"; do sleep 120; done
    onszary bash -c \
        "source ~/venvs/immu-vis/bin/activate; python3 scripts/szary/loo_aggregate.py $E f256_noact" \
        || echo "[$(stamp)] e$E aggregation failed" >> $SERIES
done
echo "[$(stamp)] series complete for epochs: $EPOCHS" >> $SERIES

#!/bin/bash
#SBATCH --account=plgspacelet2-gpu-a100
#SBATCH --job-name=val-f256-fourier
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=00-12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --output=logs/val_f256_fourier_%j.out
#SBATCH --error=logs/val_f256_fourier_%j.err

# Zero-shot hn LOO for every pre-registered epoch checkpoint of
# equiv-v3-f256-fourier-zs-hn that exists and has no result yet (e29, e49, e79,
# then every 10 epochs). Chained after each training link (afterany), so results
# appear as checkpoints land. Protocol identical to scripts/run_validation_zs_hn_equiv.sh:
# 128 px, patch_mask_ratio 0, all-channels LOO, per-image full.csv.
BASE=/net/tscratch/people/plgslukasik/immu-vis/multiplex-image-model
RUN=equiv-v3-f256-fourier-zs-hn
CK=$BASE/checkpoints_equiv_zs_hn_f256_fourier
EPOCHS="9 19 29 49 79 89 99 109 119 129 139 149 159 169 179 189 199"
for e in $EPOCHS; do
    ck=$CK/checkpoint-$RUN-epoch_$e.pth
    out=$BASE/validation_results_zs_hn_f256_fourier_e${e}_r0.0
    [ -f "$ck" ] || continue
    ls "$out"/*_full.csv >/dev/null 2>&1 && continue
    echo "######## epoch $e ########"
    CFG=configs/train_equivariant_v3_ddp_zs_hn_f256_fourier_config.yaml CKPT=$ck \
        TAG=f256_fourier_e$e WAIT_FOR_CKPT=600 \
        bash $BASE/scripts/run_validation_zs_hn_equiv.sh || echo "eval e$e FAILED"
done
echo "pending evals done"

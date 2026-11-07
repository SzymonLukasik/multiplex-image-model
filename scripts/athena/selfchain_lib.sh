# Self-resubmitting training links, so a long run never holds more than ONE job in
# the queue (shared grant: keep our footprint at 2 jobs total, one per run).
#
# Instead of pre-queueing a dependency chain, each link asks SLURM for a signal
# 15 min before its wall (#SBATCH --signal=B:USR1@900) and only then submits its
# successor (afterany on itself). The successor is therefore pending only during
# those last 15 minutes. No successor is submitted when
#   - the run is finished ($CHAIN_DONE_FILE exists, i.e. final_model was written),
#   - the link budget is used up (LINK >= CHAIN_MAX_LINKS), or
#   - training exited on its own (finished or crashed: no signal, no resubmit,
#     so a crash cannot loop).
#
# Usage, in the batch script:
#   CHAIN_SCRIPT=<absolute path of this batch script>
#   CHAIN_DONE_FILE=<final_model path>
#   CHAIN_MAX_LINKS=<n>
#   source scripts/selfchain_lib.sh
#   chain_run srun torchrun ...
LINK=${LINK:-1}

chain_resubmit() {
    echo "[chain] USR1 at $(date) (link $LINK/$CHAIN_MAX_LINKS)"
    if [ -f "$CHAIN_DONE_FILE" ]; then
        echo "[chain] $CHAIN_DONE_FILE exists -> run finished, not resubmitting"; return
    fi
    if [ "$LINK" -ge "$CHAIN_MAX_LINKS" ]; then
        echo "[chain] link budget used up -> not resubmitting"; return
    fi
    # sbatch from inside a job inherits SLURM_* variables that can clash with the
    # new job's own options (e.g. SLURM_MEM_PER_NODE), so drop them first.
    ( for v in $(compgen -e | grep '^SLURM_'); do unset "$v"; done
      cd /net/tscratch/people/plgslukasik/immu-vis/multiplex-image-model
      sbatch --dependency=afterany:$CHAIN_JOB_ID --export=ALL,LINK=$((LINK + 1)) "$CHAIN_SCRIPT" )
}

chain_run() {
    CHAIN_JOB_ID=$SLURM_JOB_ID
    trap chain_resubmit USR1
    "$@" &
    local pid=$! rc
    # `wait` returns early (rc > 128) when the trap fires; keep waiting until the
    # step itself has exited, then rc is its exit code.
    while :; do
        wait "$pid"; rc=$?
        kill -0 "$pid" 2>/dev/null || break
    done
    echo "[chain] training step exited rc=$rc at $(date)"
    return $rc
}

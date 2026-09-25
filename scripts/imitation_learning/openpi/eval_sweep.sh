#!/usr/bin/env bash
# Evaluate several training checkpoints in sequence and write one summary table.
#
# For each checkpoint: fetch params+assets from OCI, serve it, run N Isaac rollouts against it,
# record the success rate, then tear everything down before the next one.
#
# The policy server runs on CPU (JAX_PLATFORMS=cpu) so the GPU is left entirely to Isaac Sim.
# A GPU-served policy needs ~9 GB, Isaac needs ~16.5 GB, and with another training job already
# holding ~12 GB on this 32 GB card the three do not fit -- the symptom is an opaque
# "Failed to get DOF velocities from backend" from Isaac, not an out-of-memory message.
# CPU inference costs ~6.4 s/call, i.e. ~11 min per 800-step rollout at --replan_steps 8.
#
# Usage:
#   scripts/imitation_learning/openpi/eval_sweep.sh 5000 10000 15000 19999
#   ROLLOUTS=4 HORIZON=400 scripts/imitation_learning/openpi/eval_sweep.sh 5000
set -uo pipefail

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && { echo "usage: eval_sweep.sh <step> [step ...]" >&2; exit 2; }

ROLLOUTS="${ROLLOUTS:-8}"
HORIZON="${HORIZON:-800}"
REPLAN="${REPLAN:-8}"
PORT="${PORT:-8010}"
REPO=/home/benjamin.li/src/SoftMimicGen
CKPT_ROOT="${CKPT_ROOT:-/tmp/smg_eval_ckpt}"
OUT_DIR="${OUT_DIR:-$REPO/outputs/eval}"
SUMMARY="$OUT_DIR/sweep_summary.txt"
OPENPI_PY=/home/benjamin.li/src/openpi/.venv/bin/python
ISAAC_PY=/home/benjamin.li/miniforge3/envs/softmimicgen/bin/python
KEEP_CKPT="${KEEP_CKPT:-0}"

mkdir -p "$OUT_DIR" "$CKPT_ROOT"
{
    echo "# checkpoint sweep started $(date -Is)"
    echo "# rollouts=$ROLLOUTS horizon=$HORIZON replan=$REPLAN (policy served on CPU)"
    printf '# %-8s %-10s %-10s %s\n' step success rollouts note
} >> "$SUMMARY"

# Kill by explicit PID, never `pkill -f <pattern>`: the pattern matches this script's own
# command line and has silently killed the launcher before.
kill_pid() {
    local pid="$1"
    [ -z "$pid" ] && return 0
    kill -TERM "$pid" 2>/dev/null
    sleep 5
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
    return 0
}

for step in "${STEPS[@]}"; do
    echo "=============== step $step ==============="
    log_prefix="$OUT_DIR/step_${step}"

    if [ ! -d "$CKPT_ROOT/$step/params" ]; then
        echo "[$step] fetching ..."
        if ! "$REPO/scripts/imitation_learning/openpi/fetch_checkpoint.sh" "$step" "$CKPT_ROOT" \
                > "${log_prefix}_fetch.log" 2>&1; then
            printf '  %-8s %-10s %-10s %s\n' "$step" "-" "0" "FETCH FAILED" >> "$SUMMARY"
            echo "[$step] fetch failed, skipping"
            continue
        fi
    fi

    echo "[$step] serving on CPU ..."
    JAX_PLATFORMS=cpu nohup "$OPENPI_PY" \
        "$REPO/scripts/imitation_learning/openpi/serve_checkpoint.py" \
        --checkpoint_dir "$CKPT_ROOT/$step" --port "$PORT" \
        > "${log_prefix}_serve.log" 2>&1 &
    serve_pid=$!

    ready=0
    for _ in $(seq 1 60); do
        grep -q "server listening" "${log_prefix}_serve.log" 2>/dev/null && { ready=1; break; }
        kill -0 "$serve_pid" 2>/dev/null || break
        sleep 5
    done
    if [ "$ready" -ne 1 ]; then
        printf '  %-8s %-10s %-10s %s\n' "$step" "-" "0" "SERVER FAILED TO START" >> "$SUMMARY"
        kill_pid "$serve_pid"
        continue
    fi

    echo "[$step] running $ROLLOUTS rollouts ..."
    OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_PY" -u \
        "$REPO/scripts/imitation_learning/openpi/eval_policy.py" \
        --task Isaac-Fold-Towel-Yam-Joint-v0 --policy remote --port "$PORT" \
        --horizon "$HORIZON" --num_rollouts "$ROLLOUTS" --replan_steps "$REPLAN" \
        --enable_cameras --headless > "${log_prefix}_eval.log" 2>&1
    eval_rc=$?

    rate=$(grep -aoE "Success rate: [0-9.]+" "${log_prefix}_eval.log" | tail -1 | awk '{print $3}')
    trials=$(grep -acE "^\[INFO\] Trial [0-9]+: success=" "${log_prefix}_eval.log")
    note="ok"
    [ -z "$rate" ] && { rate="-"; note="NO RESULT (rc=$eval_rc)"; }
    printf '  %-8s %-10s %-10s %s\n' "$step" "$rate" "${trials:-0}" "$note" >> "$SUMMARY"
    echo "[$step] success rate=$rate over ${trials:-0} trials"

    kill_pid "$serve_pid"
    # Isaac frequently survives its own exit; leaving it running would starve the next rollout.
    for p in $(pgrep -f "openpi/eval_policy.py" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
    sleep 10
    [ "$KEEP_CKPT" = "0" ] && rm -rf "${CKPT_ROOT:?}/$step"
done

echo "# sweep finished $(date -Is)" >> "$SUMMARY"
echo
echo "===== SUMMARY ====="
cat "$SUMMARY"

#!/usr/bin/env bash
# Validate CLOSED (Schrodinger |psi>) checkpoint/restart on the -load_model path, bit-identically.
#
# Methodology (same as the open cluster/ckpt gate): with a checkpoint written every `every` steps, a
# run that completes normally still leaves a mid-run checkpoint on disk; re-issuing the SAME command
# then AUTO-RESUMES from it. Correct restart => the resumed result is byte-identical to a run that
# never checkpointed.
#   A = no checkpoint            (ground truth)
#   B = checkpoint every `every` (completes; leaves a step-`every` checkpoint)
#   C = SAME command as B        (must resume from B's checkpoint)
#   PASS iff  B == A  (writing doesn't perturb)  and  C == A  (resume is bit-identical).
#
# Works single-rank (-results_out) and multi-rank (per-rank propcheck shards -> merge_ranks.py).
# Requires -tc_integrator 1 (checkpointing is wired through the custom RK4).
#
# Usage:
#   source setup_env.sh
#   validation/validate_closed_checkpoint.sh <run_sim> <closed_model.json> <ckpt_every> [nranks]
#
# Example:
#   validation/validate_closed_checkpoint.sh ./build_local/run_sim /path/model_4q4c_closed.json 20000 4
set -euo pipefail

if [[ $# -lt 3 ]]; then grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 2; fi
BIN=$1; MODEL=$2; EVERY=$3; NRANKS=${4:-1}
[[ -x "$BIN" ]]   || { echo "not executable: $BIN" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "no model: $MODEL"   >&2; exit 2; }
command -v mpiexec >/dev/null 2>&1 || { echo "mpiexec not on PATH -- source setup_env.sh" >&2; exit 2; }

WORK=$(mktemp -d); CK="$WORK/ck"; mkdir -p "$CK"
STAMP="cck_$$"
cleanup(){ rm -rf "$WORK" propcheck/"${STAMP}"_*_rank*.txt 2>/dev/null || true; }
trap cleanup EXIT

STEPS=$(python3 -c "import json;print(json.load(open('$MODEL'))['evolution']['num_steps'])")
echo "== closed checkpoint validation: $(basename "$MODEL")  steps=$STEPS  every=$EVERY  nranks=$NRANKS =="

# run the model once; capture its final dump into $WORK/<name>.txt (merged if multi-rank). The tag
# drives BOTH the per-rank shard files and (with -tc_checkpoint_dir) the checkpoint path, so B and C
# share a tag => C resumes B's leftover checkpoint. A uses its own tag.
run() {  # run <name> <tag> <extra-args...>
    local name=$1; local tag=$2; shift 2
    if (( NRANKS == 1 )); then
        "$BIN" -tc_integrator 1 -load_model "$MODEL" -results_out "$WORK/$name.txt" -dump_tag "$tag" "$@" \
            > "$WORK/$name.log" 2>&1 || { echo "run $name FAILED"; tail -15 "$WORK/$name.log"; exit 1; }
    else
        rm -f propcheck/"${tag}"_rank*.txt          # fresh shard dumps (checkpoint in $CK is untouched)
        mpiexec -n "$NRANKS" "$BIN" -tc_integrator 1 -load_model "$MODEL" -dump_tag "$tag" "$@" \
            > "$WORK/$name.log" 2>&1 || { echo "run $name FAILED"; tail -15 "$WORK/$name.log"; exit 1; }
        python3 validation/merge_ranks.py "propcheck/${tag}" "$WORK/$name.txt" "$NRANKS" >/dev/null
    fi
}

echo "-- A: no checkpoint --";            run A "${STAMP}_A"
echo "-- B: checkpoint every $EVERY --";  run B "${STAMP}_BC" -tc_checkpoint_every "$EVERY" -tc_checkpoint_dir "$CK"
echo "-- C: same command (resume) --";    run C "${STAMP}_BC" -tc_checkpoint_every "$EVERY" -tc_checkpoint_dir "$CK"
# B and C share tag ${STAMP}_BC + $CK so C finds B's leftover checkpoint; B.txt is merged before C runs.

pass=1
if diff -q "$WORK/A.txt" "$WORK/B.txt" >/dev/null; then echo "[PASS] B == A  (checkpoint write does not perturb)"
else echo "[FAIL] B != A"; diff "$WORK/A.txt" "$WORK/B.txt" | head; pass=0; fi
if diff -q "$WORK/A.txt" "$WORK/C.txt" >/dev/null; then echo "[PASS] C == A  (resume is bit-identical)"
else echo "[FAIL] C != A"; diff "$WORK/A.txt" "$WORK/C.txt" | head; pass=0; fi
if grep -qi "RESUMING from step" "$WORK/C.log"; then
    echo "[PASS] C took the resume path: $(grep -ihm1 'RESUMING from step' "$WORK/C.log" | sed 's/.*RESUMING/RESUMING/')"
else echo "[WARN] no RESUMING log line in C (check would-be resume)"; fi

[[ $pass == 1 ]] && { echo "OVERALL: PASS"; exit 0; } || { echo "OVERALL: FAIL"; exit 1; }

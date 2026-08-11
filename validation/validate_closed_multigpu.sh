#!/usr/bin/env bash
# Validate the CLOSED (Schrodinger |psi>) multi-GPU state-sharding path.
#
# Proves that an N-rank MPI-sharded run reconstructs the SAME final states as a single-process run,
# byte for byte: single-process dump == concat(rank shards) merged into global order. This is the
# closed-system analogue of the open density-matrix sharding gate; it exercises exactly the pieces a
# real multi-node run uses -- node-local device binding, shard_indices over the 2^NQ computational
# kets, per-rank propcheck/<tag>_rank<r>.txt dumps, and validation/merge_ranks.py -- on the
# self-contained -dump_tc -tc_schrodinger -tc_computational route (no external model.json needed).
#
# On a single physical GPU every rank binds GPU 0 and shares it; the ket evolution per rank is
# independent, so bit-identicality still holds and is the correctness gate. On a real multi-GPU /
# multi-node system the same command binds one distinct GPU per rank (node-local by hostname).
#
# Usage:
#   source setup_env.sh            # puts mpiexec + PETSc/CUDA libs on PATH (needed once per shell)
#   validation/validate_closed_multigpu.sh <run_sim> <NQ> <NC> <d> <steps> <nranks> [extra flags...]
#
# Example (4Q4C, rotating driven, 4 GPUs' worth of shards):
#   validation/validate_closed_multigpu.sh ./build_local/run_sim 4 4 3 200 4 -tc_rotating_frame 1 -tc_drives 1
#
# nranks must be <= 2^NQ (the number of computational kets); the solver aborts otherwise.
set -euo pipefail

if [[ $# -lt 6 ]]; then
    grep '^#' "$0" | sed 's/^# \{0,1\}//'   # print the header as help
    exit 2
fi

BIN=$1; NQ=$2; NC=$3; D=$4; STEPS=$5; NRANKS=$6; shift 6
EXTRA=("$@")

if [[ ! -x "$BIN" ]]; then echo "validate_closed_multigpu: '$BIN' not executable" >&2; exit 2; fi
if ! command -v mpiexec >/dev/null 2>&1; then
    echo "validate_closed_multigpu: mpiexec not on PATH -- 'source setup_env.sh' first" >&2; exit 2
fi

NKETS=$(( 1 << NQ ))
if (( NRANKS > NKETS )); then
    echo "validate_closed_multigpu: nranks=$NRANKS > 2^NQ=$NKETS computational kets; pick nranks <= $NKETS" >&2
    exit 2
fi

WORK=$(mktemp -d)
STAMP="cmv_$$"
SP_TAG="${STAMP}_single"
MR_TAG="${STAMP}_shard"
COMMON=(-dump_tc -tc_schrodinger -tc_computational
        -tc_num_qubits "$NQ" -tc_num_couplers "$NC" -dump_cutoff_dim "$D"
        -dump_steps "$STEPS" "${EXTRA[@]}")

cleanup() { rm -rf "$WORK" propcheck/"${SP_TAG}"_rank*.txt propcheck/"${MR_TAG}"_rank*.txt 2>/dev/null || true; }
trap cleanup EXIT

echo "== closed multi-GPU validation: ${NQ}Q${NC}C d=${D}, ${STEPS} steps, ${NKETS} kets, ${NRANKS} ranks =="
echo "   binary: $BIN"
echo "   extra flags: ${EXTRA[*]:-(none)}"

# ---- single-process reference (mpi_size==1 -> writes the dump to stdout) --------------------------
echo "-- single-process reference --"
"$BIN" "${COMMON[@]}" -dump_tag "$SP_TAG" > "$WORK/single_raw.txt" 2> "$WORK/single.err" || {
    echo "single-process run FAILED (exit $?); stderr:"; tail -20 "$WORK/single.err"; exit 1; }
awk '/PROPAGATOR_DUMP_BEGIN/{p=1} p{print} /PROPAGATOR_DUMP_END/{p=0}' \
    "$WORK/single_raw.txt" > "$WORK/single_dump.txt"
if ! grep -q PROPAGATOR_DUMP_BEGIN "$WORK/single_dump.txt"; then
    echo "single-process produced no dump; stderr:"; tail -20 "$WORK/single.err"; exit 1
fi

# ---- N-rank sharded run (writes propcheck/<tag>_rank<r>.txt) --------------------------------------
echo "-- ${NRANKS}-rank sharded run --"
rm -f propcheck/"${MR_TAG}"_rank*.txt
mpiexec -n "$NRANKS" "$BIN" "${COMMON[@]}" -dump_tag "$MR_TAG" \
    > "$WORK/shard.out" 2> "$WORK/shard.err" || {
    echo "sharded run FAILED (exit $?); stderr:"; tail -20 "$WORK/shard.err"; exit 1; }
grep -E "evolving|wrote" "$WORK/shard.err" | sed 's/^/   /' || true

# ---- merge shards -> global order, then byte-compare ---------------------------------------------
echo "-- merge + compare --"
python3 validation/merge_ranks.py "propcheck/${MR_TAG}" "$WORK/merged.txt" "$NRANKS" "$NKETS"

if diff -q "$WORK/single_dump.txt" "$WORK/merged.txt" >/dev/null; then
    echo "RESULT: BIT-IDENTICAL  PASS  (single-process == ${NRANKS}-rank merged)"
    exit 0
else
    echo "RESULT: DIFFER  FAIL"
    diff "$WORK/single_dump.txt" "$WORK/merged.txt" | head -30
    exit 1
fi

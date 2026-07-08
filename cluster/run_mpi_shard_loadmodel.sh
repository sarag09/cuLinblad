#!/bin/bash
# MPI multi-GPU state-sharding for the -load_model (Julia handoff) path.
# Shards a loaded model's independent states across NRANKS ranks/GPUs (one shard per rank), merges
# the per-rank dumps back to global order, and -- in "validate" mode -- asserts the merged result is
# BIT-IDENTICAL to a single-process run. Run from the repo root.
#
# Usage:  cluster/run_mpi_shard_loadmodel.sh <model.json> <NRANKS> <N_STATES> <D> [MODE]
#   model.json : an open (lindblad) or closed (schrodinger) model exported by the 4Q4C Julia bridge
#   NRANKS     : MPI ranks = GPUs (request_gpus in the .sub must match)
#   N_STATES   : total evolved states -- open computational: (2^NQ)^2 ; closed computational: 2^NQ ;
#                explicit: number of dressed kets in the JSON
#   D          : hilbert_dim = prod(local_dims)   (243 for 3Q2C, 6561 for 4Q4C)
#   MODE       : "validate" => also run single-process + bit-identical gate. Feasible ONLY when the
#                              WHOLE batch fits one GPU (small models). The 4Q4C N=256 anchor
#                              (~176 GB single-process) CANNOT be bit-identical-validated -- use "run".
#                "run"      => shard + merge only (the production anchor); validate by sub-shard
#                              cross-check (compare a 2-rank vs 4-rank merge of the same model).
#
# Examples:
#   cluster/run_mpi_shard_loadmodel.sh model_3q2c_open.json 3 64 243 validate   # fast bit-identical check
#   cluster/run_mpi_shard_loadmodel.sh model_4q4c_open.json 8 256 6561 run      # the real anchor
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
source setup_env.sh
PY=${PY:-python}   # override with PY=/path/to/qutip-venv/bin/python

MODEL="${1:?model.json}"; NRANKS="${2:-2}"; N="${3:?N_STATES}"; D="${4:?hilbert_dim}"; MODE="${5:-validate}"
TAG="lmshard_$(basename "$MODEL" .json)"
FLAGS="-use_gpu_aware_mpi 0 -load_model $MODEL"

echo "=== load_model shard: -n $NRANKS, model=$MODEL, N=$N states, D=$D, mode=$MODE ==="
nvidia-smi -L || true
mkdir -p propcheck
rm -f "propcheck/${TAG}_rank"*.txt

# Multi-rank sharded evolve: each rank -> its shard file propcheck/<tag>_rank<r>.txt
mpiexec -n "$NRANKS" ./build_local/run_sim $FLAGS -dump_tag "$TAG"
$PY validation/merge_ranks.py "propcheck/$TAG" "propcheck/${TAG}_merged.txt" "$NRANKS" "$N"

if [ "$MODE" = "validate" ]; then
    # Bit-identical gate: single-process reference vs the merged shard (must be 0.0). Lindblad only,
    # and only for models whose full batch fits one GPU (NOT the 4Q4C N=256 anchor).
    echo "--- single-process reference ---"
    ./build_local/run_sim $FLAGS -results_out "propcheck/${TAG}_single.txt"
    echo "--- bit-identical check (merged vs single-process) ---"
    $PY validation/golden_regression.py "propcheck/${TAG}_merged.txt" "propcheck/${TAG}_single.txt" "$D"
fi
echo "=== done: propcheck/${TAG}_merged.txt ==="

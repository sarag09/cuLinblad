#!/bin/bash
# MPI one-rank-per-GPU state-sharding run for the open Lindblad anchor.
# Each rank binds GPU = rank%ndev, evolves a disjoint shard of the computational states, and writes
# propcheck/<tag>_rank<r>.txt; merge_ranks.py reconstructs the global-ordered dump.
#
# Usage:  run_mpi_shard.sh <NRANKS> <NQ> <NC> <STEPS> <MODE>
#   NRANKS : number of MPI ranks = number of GPUs (request_gpus in the .sub must match)
#   NQ NC  : qubits / couplers (4 4 = the production 4Q4C anchor; 3 2 = fast confirmation)
#   STEPS  : RK4 steps (1000 = converged anchor; 200 = quick check)
#   MODE   : "validate" => also run single-process + golden_regression bit-identical check
#            "run"      => multi-rank anchor only (the expensive production run)
# Examples:
#   ./run_mpi_shard.sh 2 3 2 200 validate    # 2-GPU bit-identical confirmation (fast, 3Q2C)
#   ./run_mpi_shard.sh 2 4 4 1000 run        # the real 4Q4C 256-state anchor on 2 GPUs
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
source setup_env.sh
PY=${PY:-python}   # override with PY=/path/to/qutip-venv/bin/python

NRANKS=${1:-2}
NQ=${2:-3}
NC=${3:-2}
STEPS=${4:-200}
MODE=${5:-validate}
TAG="mpishard_${NQ}q${NC}c"
D=$($PY -c "print(3**($NQ+$NC))")
# Expected merged state count: computational density propagator = (2^NQ)^2 columns.
N=$($PY -c "print((2**$NQ)**2)")

FLAGS="-use_gpu_aware_mpi 0 -dump_tc 1 -tc_num_qubits $NQ -tc_num_couplers $NC \
-dump_cutoff_dim 3 -tc_computational 1 -tc_rotating_frame 1 -tc_integrator 1 \
-tc_fused_diagonal 1 -tc_fuse_coupling 1 -tc_fused_diag_prepass 1 -dump_steps $STEPS"

echo "=== MPI shard: -n $NRANKS, ${NQ}Q${NC}C (D=$D), $STEPS steps, mode=$MODE ==="
nvidia-smi -L || true
mkdir -p propcheck
rm -f "propcheck/${TAG}_rank"*.txt

# Multi-rank sharded evolve (each rank -> GPU rank%ndev, writes its own file)
mpiexec -n "$NRANKS" ./build_local/run_sim $FLAGS -dump_tag "$TAG"
$PY validation/merge_ranks.py "propcheck/$TAG" "propcheck/${TAG}_merged.txt" "$NRANKS" "$N"

if [ "$MODE" = "validate" ]; then
    # Bit-identical gate: single-process reference vs the merged shard (must be 0.0)
    ./build_local/run_sim $FLAGS > "propcheck/${TAG}_single.txt" 2>/dev/null
    echo "--- bit-identical check (merged vs single-process) ---"
    $PY validation/golden_regression.py "propcheck/${TAG}_merged.txt" "propcheck/${TAG}_single.txt" "$D"
fi
echo "=== done: propcheck/${TAG}_merged.txt ==="

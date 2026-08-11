#!/bin/bash
# MPI one-rank-per-GPU state-sharding run for the CLOSED (Schrodinger |psi>) tunable-coupler benchmark.
#
# Closed-system multi-GPU is embarrassingly parallel: the 2^NQ computational-basis kets are independent,
# so each rank binds one GPU (node-local by hostname, across nodes too), evolves a disjoint contiguous
# slice of the kets as its batch, and writes propcheck/<tag>_rank<r>.txt; merge_ranks.py reconstructs the
# global-ordered |psi(T)> dump. No inter-rank communication during evolution. Parametric -- the
# Hamiltonian H(t) is generated in C++ (no model.json needed), so 6Q7C / 8Q10C drop in by changing NQ/NC.
#
# Usage:  run_closed_shard.sh <NRANKS> <NQ> <NC> <STEPS> [PRECISION] [MODE]
#   NRANKS    : MPI ranks = GPUs (request_gpus in the .sub must match). MUST be <= 2^NQ (the ket count).
#   NQ NC     : qubits / couplers  (4 4 = 4Q4C, 6 7 = 6Q7C, 8 10 = 8Q10C)
#   STEPS     : RK4 steps (-dump_steps)
#   PRECISION : fp64 (default, build_local) | fp32 (build_local_fp32, ~1/2 the memory)
#   MODE      : run (default; shard + merge)  |  validate (also run single-process + assert BIT-IDENTICAL;
#               feasible ONLY when the whole ket batch fits one GPU -- fine for 4Q4C, not 8Q10C fp64)
#
# Scale ceiling & per-ket memory (cutoff d=3, custom RK4 -tc_integrator 1, ~5 working state vectors):
#   system  D=3^(NQ+NC)   kets=2^NQ   ~mem/ket fp64   ~mem/ket fp32   max useful ranks
#   4Q4C    6561          16          ~0.5 MB         ~0.25 MB        16   (fits 1 GPU; multi-GPU optional)
#   6Q7C    1,594,323     64          ~0.12 GB        ~0.06 GB        64
#   8Q10C   387,420,489   256         ~31 GB          ~16 GB          256  (1 ket/GPU fp64 => ~256 GPUs)
# Pick NRANKS so ceil(2^NQ / NRANKS) kets * mem/ket fits one GPU. Use fp32 to roughly halve it.
#
# Examples:
#   ./run_closed_shard.sh 4 4 4 200 fp64 validate   # 4Q4C bit-identical multi-GPU confirmation (fast)
#   ./run_closed_shard.sh 8 4 4 12000 fp32 run      # 4Q4C closed gate, fp32, 8 GPUs
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
source setup_env.sh
PY=${PY:-python}

NRANKS=${1:-2}; NQ=${2:-4}; NC=${3:-4}; STEPS=${4:-200}; PREC=${5:-fp64}; MODE=${6:-run}

if [ "$PREC" = "fp32" ]; then
    BIN=./build_local_fp32/run_sim
    # single-precision PETSc (complex-single) must be first on the runtime lib path
    export LD_LIBRARY_PATH="$PETSC_DIR/arch-cuda-complex-single/lib:$CUDA_MERGED/lib64:${LD_LIBRARY_PATH:-}"
else
    BIN=./build_local/run_sim
fi
[ -x "$BIN" ] || { echo "no binary: $BIN (build it first: ./build.sh / ./build_fp32.sh)"; exit 1; }

D=$($PY -c "print(3**($NQ+$NC))")
NKETS=$($PY -c "print(2**$NQ)")
if [ "$NRANKS" -gt "$NKETS" ]; then
    echo "NRANKS=$NRANKS > 2^NQ=$NKETS computational kets -- ket-sharding caps at $NKETS ranks"; exit 1
fi
TAG="closedshard_${NQ}q${NC}c_${PREC}"

# Closed-system flags: -tc_schrodinger flips the benchmark into |psi> mode; -tc_computational selects the
# 2^NQ kets; -tc_integrator 1 = custom RK4 (fewer registers, required for the sizing above);
# -tc_fused_psi_diagonal fuses the single-site-diagonal RHS pass. Rotating frame + drives = the gate.
FLAGS="-use_gpu_aware_mpi 0 -dump_tc 1 -tc_schrodinger 1 -tc_computational 1 \
-tc_num_qubits $NQ -tc_num_couplers $NC -dump_cutoff_dim 3 -tc_drives 1 -tc_rotating_frame 1 \
-tc_integrator 1 -tc_fused_psi_diagonal 1 -dump_steps $STEPS"

echo "=== CLOSED shard: -n $NRANKS, ${NQ}Q${NC}C (D=$D, $NKETS kets), $STEPS steps, $PREC, mode=$MODE ==="
nvidia-smi -L || true
mkdir -p propcheck
rm -f "propcheck/${TAG}_rank"*.txt

# Multi-rank sharded evolve (each rank -> its node-local GPU, writes its own |psi> shard file).
mpiexec -n "$NRANKS" "$BIN" $FLAGS -dump_tag "$TAG"
$PY validation/merge_ranks.py "propcheck/$TAG" "propcheck/${TAG}_merged.txt" "$NRANKS" "$NKETS"

if [ "$MODE" = "validate" ]; then
    # Bit-identical gate: single-process reference (dump block from stdout) vs the merged shard.
    echo "--- single-process reference + bit-identical check ---"
    "$BIN" $FLAGS -dump_tag "${TAG}_sp" 2>/dev/null \
        | awk '/PROPAGATOR_DUMP_BEGIN/{p=1} p{print} /PROPAGATOR_DUMP_END/{p=0}' \
        > "propcheck/${TAG}_single.txt"
    if diff -q "propcheck/${TAG}_single.txt" "propcheck/${TAG}_merged.txt" >/dev/null; then
        echo "BIT-IDENTICAL  PASS  (single-process == ${NRANKS}-rank merged)"
    else
        echo "DIFFER  FAIL"; diff "propcheck/${TAG}_single.txt" "propcheck/${TAG}_merged.txt" | head -20; exit 1
    fi
fi
echo "=== done: propcheck/${TAG}_merged.txt ==="

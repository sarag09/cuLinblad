#!/bin/bash
# Multi-node GPU-binding validation -- runnable on ONE GPU (interactive OR batch).
# Builds run_sim, then exercises the new node-local-rank binding + global sharding:
#   T1/T2 : 2 ranks SHARE the 1 GPU, evolve a 64-state (3Q2C) shard each, merged == single-process
#           (bit-identical 0.0) -> sharding correctness through the new code path.
#   T3    : MULTI-NODE SIMULATION via the test knobs -- TC_FAKE_RANKS_PER_NODE=2 + TC_FAKE_GPUS_PER_NODE=2
#           makes 4 ranks look like 2 nodes x 2 GPUs. The per-rank log should show intended GPU
#           0,1,0,1 (real bound GPU 0 on this single device); 4-way merge still bit-identical 0.0.
# The real 2-PHYSICAL-GPU distinct binding is a separate request_gpus=2 batch job (mpi_shard.sub).
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
source setup_env.sh
echo "############ BUILD (CUDA arch=$CULINDBLAD_CUDA_ARCH) ############"
./build.sh
echo "############ GPUs visible on $(hostname) ############"
nvidia-smi -L || true

echo ""
echo "############ T1+T2: 2 ranks share 1 GPU, 3Q2C 64-state shard, bit-identical vs single-process ############"
cluster/run_mpi_shard.sh 2 3 2 200 validate

echo ""
echo "############ T3: MULTI-NODE SIM -- fake 2 nodes x 2 GPUs (4 ranks) on this 1 GPU ############"
echo "   EXPECT: per-rank log shows 'intended GPU 0,1,0,1' (bound GPU 0), and bit-identical == 0.000e+00"
export TC_FAKE_RANKS_PER_NODE=2 TC_FAKE_GPUS_PER_NODE=2
cluster/run_mpi_shard.sh 4 3 2 200 validate
unset TC_FAKE_RANKS_PER_NODE TC_FAKE_GPUS_PER_NODE
echo ""
echo "############ DONE -- all three checks above must read PASS / 0.000e+00 ############"

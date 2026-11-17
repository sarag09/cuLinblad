#!/usr/bin/env bash
# Build the fp32 solver (with checkpoint/restart) on a GPU node, then validate checkpoint+resume
# bit-identically on a small open model (e.g. 2Q1C). Writes a PASS/FAIL report to
# <repo>/ckpt_validation_results.txt. HTCondor wrapper: cluster/ckpt_validate.sub.
#
# Usage:  cluster/ckpt_build_validate.sh <open_model_200_steps.json> <same_model_400_steps.json>
#   The first model is checkpointed every 100 steps. The second is the same model with a different
#   step count; resuming it from the first model's checkpoint must be refused (guard test).
set -u
MODEL=$(cd "$(dirname "${1:?open model.json, 200 steps}")" && pwd)/$(basename "$1")
MODEL400=$(cd "$(dirname "${2:?same model, 400 steps}")" && pwd)/$(basename "$2")
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_env.sh"
RESULTS=$REPO/ckpt_validation_results.txt
BIN=$REPO/build_local_fp32/run_sim

{
  echo "=============================================================="
  echo " CKPT BUILD+VALIDATE  host=$(hostname)  date=$(date)"
  echo "=============================================================="
  nvidia-smi -L | head -1 || true

  echo "--- [1] clean fp32 build (cutensor-include + MPICH-pin fixes) ---"
  "$REPO/build_fp32.sh" clean > "$REPO/ckpt_val_build.log" 2>&1
  if [ ! -x "$BIN" ]; then echo "FP32 BUILD FAILED (see ckpt_val_build.log):"; tail -8 "$REPO/ckpt_val_build.log"; exit 1; fi
  echo "fp32 binary: $BIN  ($(stat -c %y "$BIN"))"

  # runtime libs: single-precision PETSc first (libpetsc complex-single), CUDA merged; libmpi via rpath.
  export LD_LIBRARY_PATH="$PETSC_DIR/arch-cuda-complex-single/lib:$CUDA_MERGED/lib64:${LD_LIBRARY_PATH:-}"
  WORK=$REPO/ckpt_val; rm -rf "$WORK"; mkdir -p "$WORK/ck"; cd "$WORK"
  COMMON="-tc_integrator 1 -use_gpu_aware_mpi 0 -tc_fused_diagonal 1 -tc_fuse_coupling 1 -tc_fused_diag_prepass 1"

  echo "--- [2] run A: straight, NO checkpoint (ground truth) ---"
  "$BIN" $COMMON -load_model "$MODEL" -results_out A.txt > a.log 2>&1; echo "A exit=$?"

  echo "--- [3] run B: checkpointed (every 100 steps), fresh dir ---"
  "$BIN" $COMMON -load_model "$MODEL" -results_out B.txt -dump_tag ckt \
        -tc_checkpoint_every 100 -tc_checkpoint_dir ck > b.log 2>&1; echo "B exit=$?"
  echo "  ckpt file after B:"; ls -l ck/ 2>&1 | sed 's/^/    /'
  grep -i "ckpt" b.log | sed 's/^/    B: /' | head

  echo "--- [4] run C: SAME command -> must AUTO-RESUME from step 100 ---"
  "$BIN" $COMMON -load_model "$MODEL" -results_out C.txt -dump_tag ckt \
        -tc_checkpoint_every 100 -tc_checkpoint_dir ck > c.log 2>&1; echo "C exit=$?"
  grep -i "ckpt" c.log | sed 's/^/    C: /' | head

  echo "--- [5] GUARD test: 400-step model vs the 200-step checkpoint -> must HARD-ABORT ---"
  "$BIN" $COMMON -load_model "$MODEL400" -results_out G.txt -dump_tag ckt \
        -tc_checkpoint_every 100 -tc_checkpoint_dir ck > g.log 2>&1; echo "G exit=$? (expect nonzero)"
  grep -iE "refusing to resume|num_steps mismatch|error" g.log | sed 's/^/    G: /' | head

  echo "=============================================================="
  echo " VERDICT"
  echo "=============================================================="
  pass=1
  if [ -s A.txt ] && [ -s B.txt ] && diff -q A.txt B.txt >/dev/null; then
    echo "  [PASS] B == A  (checkpoint writing does not perturb evolution)"
  else echo "  [FAIL] B != A"; diff A.txt B.txt 2>&1 | head -6; pass=0; fi
  if [ -s A.txt ] && [ -s C.txt ] && diff -q A.txt C.txt >/dev/null; then
    echo "  [PASS] C == A  (RESUME from checkpoint is bit-identical)"
  else echo "  [FAIL] C != A (resume not bit-identical)"; diff A.txt C.txt 2>&1 | head -6; pass=0; fi
  if grep -qiE "refusing to resume|mismatch" g.log; then
    echo "  [PASS] guard hard-aborts a mismatched (num_steps) checkpoint"
  else echo "  [FAIL] guard did NOT reject the mismatched checkpoint"; pass=0; fi
  if grep -qi "RESUMING from step 100" c.log; then
    echo "  [PASS] C actually took the resume path (RESUMING from step 100)"
  else echo "  [WARN] no explicit RESUMING log line in C (check c.log)"; fi
  echo
  [ "$pass" = 1 ] && echo "OVERALL: PASS" || echo "OVERALL: FAIL"
} 2>&1 | tee "$RESULTS"
echo "DONE ckpt validation $(date). Report: $RESULTS"

#!/usr/bin/env bash
# GPU OPTIMIZATION REGRESSION GATE for the driven tunable-coupler model.
#
# Re-dumps the driven computational propagator with the current build and checks it
# against two oracles:
#   (1) the frozen GOLDEN master  -> bit/rounding regression (fixed-step => deterministic;
#       a physics-level diff >=1e-6 is a bug)                    [golden_regression.py]
#   (2) the QuTiP reference        -> O(dt^4) convergence to ~1e-8 + trace/Hermiticity
#       invariants                                          [verify_tunable_coupler_driven.py]
#
# The C++ GPU path is the ONLY thing being optimized; the CPU reference is untouched and
# serves as a backend cross-check.
#
#   Usage: ./validate_gpu.sh <tag> <NQ> <NC> <d>
#   e.g.   ./validate_gpu.sh tc2q1c_drv 2 1 3      (fast: D=27)
#          ./validate_gpu.sh tc3q2c_drv 3 2 3      (heavier: D=243)
#
# Prereqs: a built build_local/run_sim, propcheck/<tag>_qutip_comp.npy (QuTiP ref), and
# propcheck/golden_<tag>_comp{1000,16000}.txt (run freeze_golden.sh on a validated build).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/setup_env.sh" >/dev/null 2>&1
PY=/nfs_scratch/kunnumbrath/qutip-venv/bin/python
tag="$1"; NQ="$2"; NC="$3"; d="$4"
D=$(( d ** (NQ + NC) ))

run() {
  ./build_local/run_sim -use_gpu_aware_mpi 0 -dump_tc 1 \
    -tc_num_qubits "$NQ" -tc_num_couplers "$NC" -dump_cutoff_dim "$d" \
    -tc_drives 1 -tc_computational 1 "$@"
}

echo "[validate] $tag (NQ=$NQ NC=$NC d=$d, D=$D): re-dumping driven computational states (gpu)"
echo "[validate]   @1000 steps ..."
time run -dump_steps 1000  > "propcheck/${tag}_comp1000.txt"  2>/dev/null
echo "[validate]   @16000 steps ..."
time run -dump_steps 16000 > "propcheck/${tag}_comp16000.txt" 2>/dev/null

rc=0
for s in 1000 16000; do
  g="propcheck/golden_${tag}_comp${s}.txt"
  if [ -f "$g" ]; then
    echo "[validate] golden regression @${s}:"
    $PY "$HERE/golden_regression.py" "propcheck/${tag}_comp${s}.txt" "$g" "$D" || rc=1
  else
    echo "[validate] (no golden for @${s}; skip — run freeze_golden.sh on a validated build)"
  fi
done

echo "[validate] QuTiP convergence + invariants:"
$PY "$HERE/verify_tunable_coupler_driven.py" "$tag" "$NQ" "$NC" "$d" | sed -n '5,40p' || rc=1

[ $rc -eq 0 ] && echo "[validate] ===== PASS =====" || echo "[validate] ===== FAIL ====="
exit $rc

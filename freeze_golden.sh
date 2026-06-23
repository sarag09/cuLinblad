#!/usr/bin/env bash
# Snapshot the current driven-computational propagator dumps as GOLDEN MASTERS.
# Run this ONCE on a fully-validated build (QuTiP-confirmed) before starting a GPU
# optimization; subsequent optimizations must reproduce these goldens bit/rounding-level
# (see validate_gpu.sh). Goldens are large and git-ignored (regenerable from a tagged
# commit).
#   Usage: ./freeze_golden.sh <tag> [steps...]     (default steps: 1000 16000)
set -e
tag="$1"; shift
steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=(1000 16000)
for s in "${steps[@]}"; do
  src="propcheck/${tag}_comp${s}.txt"
  [ -f "$src" ] || { echo "missing $src"; exit 1; }
  cp "$src" "propcheck/golden_${tag}_comp${s}.txt"
  echo "froze propcheck/golden_${tag}_comp${s}.txt ($(wc -l < "$src") lines)"
done

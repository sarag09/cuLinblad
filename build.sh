#!/usr/bin/env bash
# Build cuLindblad's run_sim against the local PETSc + HPC SDK CUDA/cuTENSOR.
# Usage:  ./build.sh            (configure if needed, then build)
#         ./build.sh clean      (wipe build_local and reconfigure)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/setup_env.sh"
BUILD="$HERE/build_local"
[ "$1" = "clean" ] && rm -rf "$BUILD"
mkdir -p "$BUILD"
cd "$BUILD"
# Reconfigure when there is no valid configure yet. Check for Makefile (only written
# on a SUCCESSFUL configure), not CMakeCache.txt (left behind by a FAILED configure).
if [ ! -f Makefile ]; then
  rm -f CMakeCache.txt 2>/dev/null   # drop any stale cache from a failed configure
  # Pin CMake's MPI to the SAME MPICH wrapper PETSc was configured with (first on PATH from setup_env);
  # otherwise find_package(MPI) may auto-detect a stray OpenMPI and petscsys.h aborts with
  # "configured with MPICH but ... non-MPICH mpi.h".
  cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="$CULINDBLAD_CUDA_ARCH" \
        -DCUDAToolkit_ROOT="$CUDA_MERGED" \
        -DCMAKE_PREFIX_PATH="$CUDA_MERGED" \
        -DMPI_C_COMPILER="$(command -v mpicc)" -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
        -DCUTENSOR_LIB="$CUDA_MERGED/lib64/libcutensor.so" \
        "$HERE"
fi
make -j"$(nproc)"
echo ""
echo "Built: $BUILD/run_sim"

#!/usr/bin/env bash
# Build the SINGLE-PRECISION (fp32) run_sim into build_local_fp32, against the single-precision complex
# PETSc arch (arch-cuda-complex-single). Leaves the fp64 build_local/run_sim untouched.
# Prereq: PETSc single arch built (petsc/arch-cuda-complex-single/lib/libpetsc.so).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/setup_env.sh"
# point the toolchain at the SINGLE-precision PETSc arch (pkg-config + runtime libs); MPI wrappers
# stay the shared MPICH already on PATH from setup_env (single arch reuses the fp64 arch's MPI).
export PETSC_ARCH=arch-cuda-complex-single
export PKG_CONFIG_PATH="$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig:${PKG_CONFIG_PATH#*:}"
export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:$CUDA_MERGED/lib64"
BUILD="$HERE/build_local_fp32"
[ "$1" = "clean" ] && rm -rf "$BUILD"
mkdir -p "$BUILD"; cd "$BUILD"
if [ ! -f Makefile ]; then
  rm -f CMakeCache.txt 2>/dev/null
  # Pin CMake's MPI to the SAME MPICH wrapper PETSc was configured with (first on PATH from setup_env);
  # otherwise find_package(MPI) may auto-detect a stray OpenMPI and petscsys.h aborts with
  # "configured with MPICH but ... non-MPICH mpi.h".
  cmake -DCMAKE_BUILD_TYPE=Release -DCULINDBLAD_FP32=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$CULINDBLAD_CUDA_ARCH" \
        -DCUDAToolkit_ROOT="$CUDA_MERGED" -DCMAKE_PREFIX_PATH="$CUDA_MERGED" \
        -DMPI_C_COMPILER="$(command -v mpicc)" -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
        -DCUTENSOR_LIB="$CUDA_MERGED/lib64/libcutensor.so" "$HERE"
fi
make -j8
echo "FP32_BUILD_DONE $BUILD/run_sim"

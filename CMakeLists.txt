cmake_minimum_required(VERSION 3.14)
project(cuLindblad LANGUAGES CXX CUDA)

cmake_policy(SET CMP0104 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(MPI REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)

find_library(CUTENSOR_LIB NAMES cutensor REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(run_sim
    cpp/src/apps/cli_main.cpp
    cpp/src/backends/cpu/apply_terms_cpu.cpp
    cpp/src/backends/cpu/backend_cpu_reference.cpp
    cpp/src/core/grouped_contraction_backend.cpp
    cpp/src/core/k_site_block_map.cpp
    cpp/src/core/k_site_contraction_api.cpp
    cpp/src/core/k_site_contraction_desc.cpp
    cpp/src/core/k_site_grouped_apply.cpp
    cpp/src/core/k_site_index_roles.cpp
    cpp/src/core/k_site_operator_embed.cpp
    cpp/src/core/k_site_plan.cpp
    cpp/src/core/k_site_tensor_view.cpp
    cpp/src/core/liouvillian_terms.cpp
    cpp/src/core/local_operator_utils.cpp
    cpp/src/core/local_term_utils.cpp
    cpp/src/petsc/petsc_apply.cpp
    cpp/src/petsc/petsc_shell.cpp
    cpp/src/petsc/petsc_ts_rhs.cpp
    cpp/src/petsc/petsc_ts_smoke.cpp
    cpp/src/backends/cuda/cutensor_contraction_desc.cpp
    cpp/src/backends/cuda/cutensor_ops.cpp
    cpp/src/backends/cuda/cutensor_tensor_descs.cpp
    cpp/src/backends/cuda/cutensor_operation_desc.cpp
    cpp/src/backends/cuda/cutensor_plan.cpp
    cpp/src/backends/cuda/cutensor_execute.cpp
    cpp/src/backends/cuda/cutensor_executor.cpp
    cpp/src/backends/cuda/cutensor_dissipator.cpp
    cpp/src/backends/cuda/cuda_elementwise.cu
    cpp/src/core/time_dependence.cpp
    cpp/src/core/time_dependent_term.cpp
    cpp/src/backends/cuda/cutensor_executor_cache.cpp
    cpp/src/backends/cuda/pinned_host_buffer.cpp
    cpp/src/core/grouped_state_layout.cpp
    cpp/src/petsc/petsc_cuda_apply.cpp
    cpp/src/petsc/petsc_cuda_ts_rhs.cpp
    cpp/src/petsc/petsc_cuda_ts_smoke.cpp
    cpp/src/backends/cuda/cuda_grouped_layout.cu
    cpp/src/petsc/petsc_cuda_vec_utils.cpp
    cpp/src/core/batch_evolution.cpp
    cpp/src/core/batched_grouped_layout.cpp
    cpp/src/backends/cuda/batched_grouped_apply.cpp
    cpp/src/core/transmon_chain_benchmark.cpp
    cpp/src/core/tunable_coupler_benchmark.cpp
    cpp/src/io/model_io.cpp
    cpp/src/io/pulse_envelope.cpp
)

target_include_directories(run_sim PRIVATE
    cpp/include
    cpp/third_party
)

# cuTENSOR headers (cutensor.h). The solver links CUTENSOR_LIB but the matching include dir was never
# declared here -- it used to be supplied via an environment include path that the paper cleanup of
# setup_env.sh dropped, and the $CUDA_MERGED/include/cutensor.h symlink can dangle per-node. Declare a
# real cuTENSOR include dir explicitly so the build is self-contained. Prefer $CUTENSOR_ROOT/include
# (a real header, no stray mpi.h to shadow the MPICH one PETSc needs); fall back to $CUDA_MERGED/include.
if(DEFINED ENV{CUTENSOR_ROOT} AND EXISTS "$ENV{CUTENSOR_ROOT}/include/cutensor.h")
    target_include_directories(run_sim PRIVATE "$ENV{CUTENSOR_ROOT}/include")
    message(STATUS "cuLindblad: cuTENSOR include = $ENV{CUTENSOR_ROOT}/include")
elseif(DEFINED ENV{CUDA_MERGED} AND EXISTS "$ENV{CUDA_MERGED}/include/cutensor.h")
    target_include_directories(run_sim PRIVATE "$ENV{CUDA_MERGED}/include")
    message(STATUS "cuLindblad: cuTENSOR include = $ENV{CUDA_MERGED}/include")
else()
    message(WARNING "cuLindblad: no cutensor.h found via CUTENSOR_ROOT/CUDA_MERGED; build may fail")
endif()

target_link_libraries(run_sim PRIVATE
    PkgConfig::PETSC
    MPI::MPI_CXX
    CUDA::cudart
    CUDA::cublas
    ${CUTENSOR_LIB}
)

# Single-precision (complex float) build. Default OFF = fp64. When ON, the solver's Complex/Real,
# the cuTENSOR tensor/compute descriptors (32F), and the CUDA-kernel device complex type all switch
# to single precision; run_sim MUST then be linked against a PETSc arch built --with-precision=single
# --with-scalar-type=complex (PetscScalar = complex float). Build into a SEPARATE dir so the fp64
# binary is untouched, e.g.:  cmake -DCULINDBLAD_FP32=ON -DPETSC_ARCH=arch-cuda-complex-single ...
option(CULINDBLAD_FP32 "Build the solver in single precision (complex float)" OFF)
if(CULINDBLAD_FP32)
    target_compile_definitions(run_sim PRIVATE CULINDBLAD_FP32)
    message(STATUS "cuLindblad: SINGLE-PRECISION (fp32) build enabled")
endif()
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
)

target_include_directories(run_sim PRIVATE
    cpp/include
)

target_link_libraries(run_sim PRIVATE
    PkgConfig::PETSC
    MPI::MPI_CXX
    CUDA::cudart
    CUDA::cublas
    ${CUTENSOR_LIB}
)
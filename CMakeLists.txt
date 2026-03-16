cmake_minimum_required(VERSION 3.14)
project(cuLindblad LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find PETSc via pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)

# Find cuTensor
find_library(CUTENSOR_LIB NAMES cutensor REQUIRED)

# Find the CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

add_executable(run_sim
    cpp/src/apps/cli_main.cpp
    cpp/src/backends/cpu/apply_terms_cpu.cpp
    cpp/src/core/liouvillian_terms.cpp
)

target_include_directories(run_sim PRIVATE
    cpp/include
)

target_link_libraries(run_sim PRIVATE
    PkgConfig::PETSC
    MPI::MPI_CXX
    CUDA::cudart
    CUDA::cublas
)

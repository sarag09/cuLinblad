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

# Fetch Pybind11 directly from GitHub
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)    

FetchContent_MakeAvailable(pybind11)

# Create a Python Module instead of an Executable
pybind11_add_module(cuLindblad_core main.cpp)

# Find the CUDA Toolkit (So C++ knows where cuda_runtime.h is)
find_package(CUDAToolkit REQUIRED)

# Link BOTH PETSc and MPI to our executable
target_link_libraries(cuLindblad_core PRIVATE PkgConfig::PETSC MPI::MPI_CXX CUDA::cudart ${CUTENSOR_LIB})
cmake_minimum_required(VERSION 3.14)
project(cuLindblad CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Find MPI 
find_package(MPI REQUIRED)

# 2. Find PETSc via pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)

add_executable(run_sim main.cpp)

# 3. Link BOTH PETSc and MPI to our executable
target_link_libraries(run_sim PkgConfig::PETSC MPI::MPI_CXX)
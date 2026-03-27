#pragma once

#include <petscts.h>

#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

PetscErrorCode run_ts_cuda_grouped_left_smoke_test(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    Index row,
    Index col,
    Complex& value_out);

PetscErrorCode run_ts_cuda_grouped_liouvillian_smoke_test(
    const Solver& solver,
    const std::vector<Complex>& h_local_op,
    const std::vector<Complex>& d_local_op,
    const std::vector<Complex>& d_local_op_dag,
    const std::vector<Complex>& d_local_op_dag_op,
    const std::vector<Index>& target_sites,
    Index row,
    Index col,
    Complex& value_out);

PetscErrorCode run_ts_cuda_static_model_liouvillian_smoke_test(
    const Solver& solver,
    Index row,
    Index col,
    Complex& value_out);    

PetscErrorCode run_ts_cuda_full_model_liouvillian_smoke_test(
    const Solver& solver,
    double start_time,
    Index row,
    Index col,
    Complex& value_out);

} // namespace culindblad
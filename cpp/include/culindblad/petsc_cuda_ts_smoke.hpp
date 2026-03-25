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

} // namespace culindblad

#pragma once

#include <petscts.h>

#include "culindblad/cutensor_executor_cache.hpp"
#include "culindblad/grouped_state_layout.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

struct PetscCudaTsRhsContext {
    const Solver* solver;
    const std::vector<Complex>* local_op;
    std::vector<Index> target_sites;
    GroupedStateLayout grouped_layout;
    CuTensorExecutorCache executor_cache;
};

PetscErrorCode ts_rhs_function_cuda_grouped_commutator(
    TS ts,
    PetscReal t,
    Vec x,
    Vec f,
    void* ctx);

} // namespace culindblad

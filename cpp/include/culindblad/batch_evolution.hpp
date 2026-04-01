#pragma once

#include <vector>

#include <petscts.h>
#include <petscvec.h>

#include "culindblad/petsc_cuda_ts_rhs.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

struct BatchEvolutionConfig {
    double t0;
    double dt;
    Index num_steps;
    Index batch_size;
};

struct BatchEvolutionResult {
    std::vector<std::vector<Complex>> final_states;
};

struct BatchEvolutionTiming {
    double elapsed_seconds;
    double gpu_elapsed_seconds;
    BatchEvolutionResult result;
};

struct CudaBatchExecutionContext {
    TS ts;
    Vec x;
    PetscCudaTsRhsContext rhs_ctx;
    bool initialized;
};

BatchEvolutionResult evolve_density_batch_cpu_reference(
    const Solver& solver,
    const std::vector<std::vector<Complex>>& initial_states,
    const BatchEvolutionConfig& config);

BatchEvolutionResult evolve_density_batch_cuda_ts(
    const Solver& solver,
    const std::vector<std::vector<Complex>>& initial_states,
    const BatchEvolutionConfig& config);

BatchEvolutionTiming time_density_batch_cpu_reference(
    const Solver& solver,
    const std::vector<std::vector<Complex>>& initial_states,
    const BatchEvolutionConfig& config);

BatchEvolutionTiming time_density_batch_cuda_ts(
    const Solver& solver,
    const std::vector<std::vector<Complex>>& initial_states,
    const BatchEvolutionConfig& config);

PetscErrorCode create_cuda_batch_execution_context(
    const Solver& solver,
    Index batch_size,
    CudaBatchExecutionContext& ctx);

PetscErrorCode destroy_cuda_batch_execution_context(
    CudaBatchExecutionContext& ctx);

} // namespace culindblad

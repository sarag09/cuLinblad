#pragma once

#include <petscts.h>

#include <cstddef>
#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "culindblad/cuda_grouped_layout.hpp"
#include "culindblad/cutensor_executor_cache.hpp"
#include "culindblad/grouped_state_layout.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

struct CachedDissipatorAuxiliaries {
    std::string name;
    std::vector<Index> sites;
    std::vector<Complex> l_op;
    std::vector<Complex> l_dag;
    std::vector<Complex> l_dag_l;
    std::vector<Complex> jump_diagonal;
    void* d_jump_diagonal = nullptr;
};

struct CachedGroupedLayoutEntry {
    std::vector<Index> sites;
    GroupedStateLayout grouped_layout;
    CudaGroupedStateLayout cuda_grouped_layout;
    std::size_t grouped_bytes = 0;
    std::vector<Complex> static_hamiltonian_sum;
    std::vector<Complex> static_dissipator_norm_sum;
};

struct GroupedRhsScratchBuffers {
    void* d_grouped_input = nullptr;
    void* d_grouped_term = nullptr;
    void* d_grouped_accum = nullptr;
    std::size_t grouped_bytes = 0;
};

struct PetscCudaTsRhsContext {
    const Solver* solver;
    Index batch_size;
    const std::vector<Complex>* h_local_op;
    const std::vector<Complex>* d_local_op;
    const std::vector<Complex>* d_local_op_dag;
    const std::vector<Complex>* d_local_op_dag_op;
    std::vector<Index> target_sites;
    GroupedStateLayout grouped_layout;
    CudaGroupedStateLayout cuda_grouped_layout;
    CuTensorExecutorCache executor_cache;
    std::vector<CachedDissipatorAuxiliaries> cached_static_dissipators;
    std::vector<CachedGroupedLayoutEntry> cached_grouped_layouts;
    GroupedRhsScratchBuffers grouped_scratch;
    cudaStream_t elementwise_stream;

    Vec work_vec_a;
    Vec work_vec_b;
    Vec work_vec_c;
};

PetscErrorCode ts_rhs_function_cuda_grouped_commutator(
    TS ts,
    PetscReal t,
    Vec x,
    Vec f,
    void* ctx);

PetscErrorCode ts_rhs_function_cuda_grouped_liouvillian(
    TS ts,
    PetscReal t,
    Vec x,
    Vec f,
    void* ctx);

PetscErrorCode ts_rhs_function_cuda_static_model_liouvillian(
    TS ts,
    PetscReal t,
    Vec x,
    Vec f,
    void* ctx);

PetscErrorCode ts_rhs_function_cuda_full_model_liouvillian(
    TS ts,
    PetscReal t,
    Vec x,
    Vec f,
    void* ctx);

} // namespace culindblad

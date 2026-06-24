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
    // Device copy of the (small, dense) collapse operator L for non-diagonal single-site
    // dissipators (e.g. relaxation a), used by the dense jump fast path.
    void* d_l_op = nullptr;
};

struct CachedGroupedLayoutEntry {
    std::vector<Index> sites;
    GroupedStateLayout grouped_layout;
    CudaGroupedStateLayout cuda_grouped_layout;
    // State-vector (ket-only, D-sized) layout sharing the same descriptors, used by the
    // Schrodinger solver's one-sided H|psi> path. Built alongside the density layout.
    CudaGroupedStateLayout psi_cuda_grouped_layout;
    bool has_psi_layout = false;
    std::size_t grouped_bytes = 0;
    std::vector<Complex> static_hamiltonian_sum;
    std::vector<Complex> static_hamiltonian_diagonal;
    std::vector<Complex> static_dissipator_norm_sum;
    std::vector<Complex> static_dissipator_norm_diagonal;
    std::vector<Complex> static_sparse_hamiltonian_values;
    std::vector<Index> static_sparse_hamiltonian_rows;
    std::vector<Index> static_sparse_hamiltonian_cols;
    void* d_static_hamiltonian_diagonal = nullptr;
    void* d_static_dissipator_norm_diagonal = nullptr;
    void* d_static_sparse_hamiltonian_values = nullptr;
    Index* d_static_sparse_hamiltonian_rows = nullptr;
    Index* d_static_sparse_hamiltonian_cols = nullptr;

    // Reusable device scratch for the DYNAMIC structured fast paths. When a layout's
    // effective Hamiltonian is time-dependent but stays diagonal (driven omega(t)*n) or
    // sparse (driven g(t)*coupling), its structured values are recomputed and re-uploaded
    // here each RHS step so the cheap diagonal/sparse commutator kernels can be used
    // instead of a general cuTENSOR contraction. Allocated lazily; freed on teardown.
    void* d_dynamic_diagonal = nullptr;
    std::size_t dynamic_diagonal_capacity = 0;
    void* d_dynamic_sparse_values = nullptr;
    Index* d_dynamic_sparse_rows = nullptr;
    Index* d_dynamic_sparse_cols = nullptr;
    std::size_t dynamic_sparse_capacity = 0;
};

struct GroupedRhsScratchBuffers {
    void* d_grouped_input = nullptr;
    void* d_grouped_term = nullptr;
    void* d_grouped_accum = nullptr;
    std::size_t grouped_bytes = 0;
    // State-vector regroup scratch (D-sized, batch*hilbert_dim). Used by the Schrodinger RHS;
    // the D^2 buffers above are left null for a state-vector context.
    void* d_psi_grouped = nullptr;
    std::size_t psi_grouped_bytes = 0;
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

// Closed-system Schrodinger RHS: d|psi>/dt = -i H|psi>, matrix-free one-sided left action.
// Same context as the Liouvillian RHS, but x/f are length batch*hilbert_dim (D) vectors and
// the dissipator/anti-commutator/right-action branches are absent.
PetscErrorCode ts_rhs_function_cuda_full_model_schrodinger(
    TS ts,
    PetscReal t,
    Vec x,
    Vec f,
    void* ctx);

} // namespace culindblad

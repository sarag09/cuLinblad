#include "culindblad/petsc_cuda_ts_rhs.hpp"

#include <petscts.h>
#include <petscvec.h>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include <vector>

#include "culindblad/cuda_elementwise.hpp"
#include "culindblad/local_operator_utils.hpp"
#include "culindblad/petsc_cuda_apply.hpp"
#include "culindblad/time_dependent_term.hpp"

namespace culindblad {

namespace {

Index rhs_total_density_dim(const PetscCudaTsRhsContext& rhs_ctx)
{
    return rhs_ctx.batch_size * rhs_ctx.solver->layout.density_dim;
}

Index rhs_total_hilbert_dim(const PetscCudaTsRhsContext& rhs_ctx)
{
    return rhs_ctx.batch_size * rhs_ctx.solver->layout.hilbert_dim;
}

bool same_sites(
    const std::vector<Index>& a,
    const std::vector<Index>& b)
{
    if (a.size() != b.size()) {
        return false;
    }

    for (Index i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }

    return true;
}

const CachedGroupedLayoutEntry* find_cached_grouped_layout(
    const PetscCudaTsRhsContext& rhs_ctx,
    const std::vector<Index>& sites)
{
    for (const CachedGroupedLayoutEntry& entry : rhs_ctx.cached_grouped_layouts) {
        if (same_sites(entry.sites, sites)) {
            return &entry;
        }
    }

    return nullptr;
}

std::string make_sites_cache_suffix(
    const std::vector<Index>& sites)
{
    std::string suffix = "sites";
    for (Index site : sites) {
        suffix += "_" + std::to_string(site);
    }
    return suffix;
}

class ScopedNvtxRange {
public:
    explicit ScopedNvtxRange(
        const std::string& label)
    {
        nvtxRangePushA(label.c_str());
    }

    ~ScopedNvtxRange()
    {
        nvtxRangePop();
    }
};

std::string make_nvtx_label(
    const char* family,
    const std::string& term_label)
{
    return std::string(family) + ":" + term_label;
}

bool can_use_static_sparse_commutator(
    const CachedGroupedLayoutEntry& layout_entry,
    bool has_dynamic_hamiltonian)
{
    return !has_dynamic_hamiltonian &&
           layout_entry.sites.size() == 2 &&
           !layout_entry.static_sparse_hamiltonian_values.empty() &&
           layout_entry.d_static_sparse_hamiltonian_values != nullptr &&
           layout_entry.d_static_sparse_hamiltonian_rows != nullptr &&
           layout_entry.d_static_sparse_hamiltonian_cols != nullptr;
}

bool can_use_static_diagonal_commutator(
    const CachedGroupedLayoutEntry& layout_entry,
    bool has_dynamic_hamiltonian)
{
    return !has_dynamic_hamiltonian &&
           !layout_entry.static_hamiltonian_diagonal.empty() &&
           layout_entry.d_static_hamiltonian_diagonal != nullptr;
}

void accumulate_operator_scaled(
    const std::vector<Complex>& term,
    double scale,
    std::vector<Complex>& accum)
{
    if (scale == 0.0) {
        return;
    }

    if (accum.empty()) {
        accum.assign(term.size(), Complex{0.0, 0.0});
    } else if (accum.size() != term.size()) {
        throw std::runtime_error(
            "accumulate_operator_scaled: mismatched operator sizes");
    }

    for (Index i = 0; i < term.size(); ++i) {
        accum[i] += scale * term[i];
    }
}

// Apply -i[H_eff(t), rho] for a layout whose TIME-DEPENDENT effective Hamiltonian is
// still structured (diagonal or sparse), using the cheap diagonal/sparse commutator
// kernels instead of a general cuTENSOR contraction. The structured values change every
// step, so they are recomputed from effective_hamiltonian and re-uploaded into reusable
// per-layout device scratch. Returns true if a fast path was used; false => caller falls
// back to the general cuTENSOR commutator. Correctness is identical to the cuTENSOR path
// (same operator, same kernels already validated for the static case).
// Build the sparse pattern of a (matrix_dim x matrix_dim) 2-site effective Hamiltonian and
// upload it to the layout's dynamic-sparse device buffers (lazy-(re)allocated). Returns nnz
// via out_nnz; false if empty or an allocation/copy fails. Shared by the dynamic structured
// commutator and the fused sparse-commutator-to-flat coupling path.
bool upload_dynamic_sparse_hamiltonian(
    CachedGroupedLayoutEntry& layout_entry,
    const std::vector<Complex>& effective_hamiltonian,
    Index matrix_dim,
    std::size_t& out_nnz,
    cudaStream_t stream)
{
    std::vector<Complex> values;
    std::vector<Index> rows;
    std::vector<Index> cols;
    for (Index row = 0; row < matrix_dim; ++row) {
        for (Index col = 0; col < matrix_dim; ++col) {
            const Complex value = effective_hamiltonian[row * matrix_dim + col];
            if (std::abs(value) == 0.0) {
                continue;
            }
            rows.push_back(row);
            cols.push_back(col);
            values.push_back(value);
        }
    }
    if (values.empty()) {
        return false;
    }
    const std::size_t nnz = values.size();
    if (layout_entry.dynamic_sparse_capacity < nnz) {
        if (layout_entry.d_dynamic_sparse_values != nullptr) {
            cudaFree(layout_entry.d_dynamic_sparse_values);
        }
        if (layout_entry.d_dynamic_sparse_rows != nullptr) {
            cudaFree(layout_entry.d_dynamic_sparse_rows);
        }
        if (layout_entry.d_dynamic_sparse_cols != nullptr) {
            cudaFree(layout_entry.d_dynamic_sparse_cols);
        }
        layout_entry.d_dynamic_sparse_values = nullptr;
        layout_entry.d_dynamic_sparse_rows = nullptr;
        layout_entry.d_dynamic_sparse_cols = nullptr;
        layout_entry.dynamic_sparse_capacity = 0;
        if (cudaMalloc(&layout_entry.d_dynamic_sparse_values, nnz * sizeof(Complex)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void**>(&layout_entry.d_dynamic_sparse_rows), nnz * sizeof(Index)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void**>(&layout_entry.d_dynamic_sparse_cols), nnz * sizeof(Index)) != cudaSuccess) {
            return false;
        }
        layout_entry.dynamic_sparse_capacity = nnz;
    }
    if (cudaMemcpyAsync(layout_entry.d_dynamic_sparse_values, values.data(),
                        nnz * sizeof(Complex), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(layout_entry.d_dynamic_sparse_rows, rows.data(),
                        nnz * sizeof(Index), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(layout_entry.d_dynamic_sparse_cols, cols.data(),
                        nnz * sizeof(Index), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return false;
    }
    // CSR row pointers over the row-major pattern (row_ptr[r] = first nnz index of row r), so the
    // Schrodinger sparse-left kernel scans only each row's nonzeros. Unused by the density kernels.
    const std::size_t row_ptr_len = static_cast<std::size_t>(matrix_dim) + 1;
    std::vector<Index> row_ptr(row_ptr_len, 0);
    for (std::size_t i = 0; i < nnz; ++i) {
        ++row_ptr[rows[i] + 1];
    }
    for (Index r = 0; r < matrix_dim; ++r) {
        row_ptr[r + 1] += row_ptr[r];
    }
    if (layout_entry.dynamic_sparse_row_ptr_capacity < row_ptr_len) {
        if (layout_entry.d_dynamic_sparse_row_ptr != nullptr) {
            cudaFree(layout_entry.d_dynamic_sparse_row_ptr);
            layout_entry.d_dynamic_sparse_row_ptr = nullptr;
        }
        if (cudaMalloc(reinterpret_cast<void**>(&layout_entry.d_dynamic_sparse_row_ptr),
                       row_ptr_len * sizeof(Index)) != cudaSuccess) {
            layout_entry.d_dynamic_sparse_row_ptr = nullptr;
            layout_entry.dynamic_sparse_row_ptr_capacity = 0;
            return false;
        }
        layout_entry.dynamic_sparse_row_ptr_capacity = row_ptr_len;
    }
    if (cudaMemcpyAsync(layout_entry.d_dynamic_sparse_row_ptr, row_ptr.data(),
                        row_ptr_len * sizeof(Index), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return false;
    }
    out_nnz = nnz;
    return true;
}

bool try_apply_dynamic_structured_commutator(
    CachedGroupedLayoutEntry& layout_entry,
    const std::vector<Complex>& effective_hamiltonian,
    const void* d_grouped_input,
    void* d_grouped_output,
    Index batch_size,
    cudaStream_t stream)
{
    const std::size_t n = effective_hamiltonian.size();
    if (n == 0) {
        return false;
    }
    const Index matrix_dim =
        static_cast<Index>(std::llround(std::sqrt(static_cast<double>(n))));
    if (static_cast<std::size_t>(matrix_dim) * static_cast<std::size_t>(matrix_dim) != n) {
        return false;
    }
    if (matrix_dim == 0 ||
        layout_entry.grouped_layout.hilbert_dim % matrix_dim != 0) {
        return false;
    }
    const Index complement_dim =
        layout_entry.grouped_layout.hilbert_dim / matrix_dim;

    // (1) Diagonal effective Hamiltonian -> diagonal commutator kernel.
    std::vector<Complex> diagonal;
    if (try_extract_local_diagonal(effective_hamiltonian, matrix_dim, diagonal)) {
        const std::size_t bytes = diagonal.size() * sizeof(Complex);
        if (layout_entry.dynamic_diagonal_capacity < diagonal.size()) {
            if (layout_entry.d_dynamic_diagonal != nullptr) {
                cudaFree(layout_entry.d_dynamic_diagonal);
                layout_entry.d_dynamic_diagonal = nullptr;
            }
            if (cudaMalloc(&layout_entry.d_dynamic_diagonal, bytes) != cudaSuccess) {
                layout_entry.d_dynamic_diagonal = nullptr;
                layout_entry.dynamic_diagonal_capacity = 0;
                return false;
            }
            layout_entry.dynamic_diagonal_capacity = diagonal.size();
        }
        if (cudaMemcpyAsync(layout_entry.d_dynamic_diagonal, diagonal.data(), bytes,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            return false;
        }
        return launch_batched_grouped_diagonal_commutator_kernel(
            layout_entry.d_dynamic_diagonal, d_grouped_input, d_grouped_output,
            matrix_dim, complement_dim, batch_size, stream);
    }

    // (2) Sparse 2-site effective Hamiltonian -> sparse commutator kernel.
    if (layout_entry.sites.size() != 2) {
        return false;
    }
    std::size_t nnz = 0;
    if (!upload_dynamic_sparse_hamiltonian(
            layout_entry, effective_hamiltonian, matrix_dim, nnz, stream)) {
        return false;
    }
    return launch_batched_grouped_sparse_commutator_kernel(
        layout_entry.d_dynamic_sparse_values, layout_entry.d_dynamic_sparse_rows,
        layout_entry.d_dynamic_sparse_cols, static_cast<Index>(nnz),
        d_grouped_input, d_grouped_output, matrix_dim, complement_dim, batch_size, stream);
}

CachedGroupedLayoutEntry* find_cached_grouped_layout_mutable(
    PetscCudaTsRhsContext& rhs_ctx,
    const std::vector<Index>& sites)
{
    for (CachedGroupedLayoutEntry& entry : rhs_ctx.cached_grouped_layouts) {
        if (same_sites(entry.sites, sites)) {
            return &entry;
        }
    }

    return nullptr;
}

PetscErrorCode zero_petsc_cuda_vec(
    Vec v,
    Index size,
    cudaStream_t stream)
{
    PetscScalar* d_v_ptr = nullptr;

#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDAGetArray(v, &d_v_ptr));
#else
    PetscCall(VecGetArray(v, &d_v_ptr));
#endif

    const cudaError_t memset_status =
        cudaMemsetAsync(
            reinterpret_cast<void*>(d_v_ptr),
            0,
            size * sizeof(Complex),
            stream);

    if (memset_status != cudaSuccess) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(VecCUDARestoreArray(v, &d_v_ptr));
#else
        PetscCall(VecRestoreArray(v, &d_v_ptr));
#endif
        return PETSC_ERR_LIB;
    }

#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDARestoreArray(v, &d_v_ptr));
#else
    PetscCall(VecRestoreArray(v, &d_v_ptr));
#endif

    return 0;
}

PetscErrorCode add_petsc_cuda_vecs(
    Vec a,
    Vec b,
    Vec out,
    Index size,
    cudaStream_t stream)
{
    const PetscScalar* d_a_ptr = nullptr;
    const PetscScalar* d_b_ptr = nullptr;
    PetscScalar* d_out_ptr = nullptr;

#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDAGetArrayRead(a, &d_a_ptr));
    PetscCall(VecCUDAGetArrayRead(b, &d_b_ptr));
    PetscCall(VecCUDAGetArray(out, &d_out_ptr));
#else
    PetscCall(VecGetArrayRead(a, &d_a_ptr));
    PetscCall(VecGetArrayRead(b, &d_b_ptr));
    PetscCall(VecGetArray(out, &d_out_ptr));
#endif

    const bool add_ok =
        launch_vector_add_kernel(
            reinterpret_cast<const void*>(d_a_ptr),
            reinterpret_cast<const void*>(d_b_ptr),
            reinterpret_cast<void*>(d_out_ptr),
            size,
            stream);

    if (!add_ok) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(VecCUDARestoreArray(out, &d_out_ptr));
        PetscCall(VecCUDARestoreArrayRead(b, &d_b_ptr));
        PetscCall(VecCUDARestoreArrayRead(a, &d_a_ptr));
#else
        PetscCall(VecRestoreArray(out, &d_out_ptr));
        PetscCall(VecRestoreArrayRead(b, &d_b_ptr));
        PetscCall(VecRestoreArrayRead(a, &d_a_ptr));
#endif
        return PETSC_ERR_LIB;
    }

#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDARestoreArray(out, &d_out_ptr));
    PetscCall(VecCUDARestoreArrayRead(b, &d_b_ptr));
    PetscCall(VecCUDARestoreArrayRead(a, &d_a_ptr));
#else
    PetscCall(VecRestoreArray(out, &d_out_ptr));
    PetscCall(VecRestoreArrayRead(b, &d_b_ptr));
    PetscCall(VecRestoreArrayRead(a, &d_a_ptr));
#endif

    return 0;
}

PetscErrorCode scale_petsc_cuda_vec(
    Vec in,
    double scale,
    Vec out,
    Index size,
    cudaStream_t stream)
{
    const PetscScalar* d_in_ptr = nullptr;
    PetscScalar* d_out_ptr = nullptr;

#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDAGetArrayRead(in, &d_in_ptr));
    PetscCall(VecCUDAGetArray(out, &d_out_ptr));
#else
    PetscCall(VecGetArrayRead(in, &d_in_ptr));
    PetscCall(VecGetArray(out, &d_out_ptr));
#endif

    const bool scale_ok =
        launch_vector_scale_kernel(
            reinterpret_cast<const void*>(d_in_ptr),
            scale,
            reinterpret_cast<void*>(d_out_ptr),
            size,
            stream);

    if (!scale_ok) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(VecCUDARestoreArray(out, &d_out_ptr));
        PetscCall(VecCUDARestoreArrayRead(in, &d_in_ptr));
#else
        PetscCall(VecRestoreArray(out, &d_out_ptr));
        PetscCall(VecRestoreArrayRead(in, &d_in_ptr));
#endif
        return PETSC_ERR_LIB;
    }

#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDARestoreArray(out, &d_out_ptr));
    PetscCall(VecCUDARestoreArrayRead(in, &d_in_ptr));
#else
    PetscCall(VecRestoreArray(out, &d_out_ptr));
    PetscCall(VecRestoreArrayRead(in, &d_in_ptr));
#endif

    return 0;
}

PetscErrorCode zero_grouped_layout_accumulator(
    const PetscCudaTsRhsContext& rhs_ctx,
    CachedGroupedLayoutEntry& layout_entry,
    Index batch_size,
    cudaStream_t stream)
{
    const bool zero_ok =
        launch_zero_batched_buffer_kernel(
            rhs_ctx.grouped_scratch.d_grouped_accum,
            batch_size * layout_entry.grouped_layout.grouped_size,
            stream);

    return zero_ok ? 0 : PETSC_ERR_LIB;
}

// Lazily allocate d_grouped_term, the scratch used ONLY by the general cuTENSOR fallback.
// The structured tunable-coupler path never reaches that fallback, so this is never called
// there and d_grouped_term stays unallocated (one full grouped buffer, ~D^2 per state,
// saved). A non-structured model allocates it on first cuTENSOR use.
PetscErrorCode ensure_grouped_term_buffer(GroupedRhsScratchBuffers& scratch)
{
    if (scratch.d_grouped_term != nullptr) {
        return 0;
    }
    if (scratch.grouped_bytes == 0) {
        return PETSC_ERR_ARG_WRONGSTATE;
    }
    if (cudaMalloc(&scratch.d_grouped_term, scratch.grouped_bytes) != cudaSuccess) {
        scratch.d_grouped_term = nullptr;
        return PETSC_ERR_MEM;
    }
    return 0;
}

// Lazily allocate d_grouped_accum (one full grouped buffer, ~D^2 per state). Under the fully-fused
// production path (-tc_fused_diagonal 1 -tc_fuse_coupling 1) every layer writes f directly via the
// to-flat kernels, so the accumulator is never touched and stays unallocated -- ~0.64 GiB/state
// saved (~25 -> ~30 density matrices on a 96 GiB GPU). Allocated on demand only when a layer falls
// through to an accumulate-then-flatten path (the fused-diagonal accum branch or the non-fused /
// cuTENSOR-fallback path). Idempotent.
PetscErrorCode ensure_grouped_accum_buffer(GroupedRhsScratchBuffers& scratch)
{
    if (scratch.d_grouped_accum != nullptr) {
        return 0;
    }
    if (scratch.grouped_bytes == 0) {
        return PETSC_ERR_ARG_WRONGSTATE;
    }
    if (cudaMalloc(&scratch.d_grouped_accum, scratch.grouped_bytes) != cudaSuccess) {
        scratch.d_grouped_accum = nullptr;
        return PETSC_ERR_MEM;
    }
    return 0;
}

PetscErrorCode accumulate_grouped_layout_term(
    const PetscCudaTsRhsContext& rhs_ctx,
    CachedGroupedLayoutEntry& layout_entry,
    double scale,
    Index batch_size,
    cudaStream_t stream)
{
    const Index grouped_elements =
        batch_size * layout_entry.grouped_layout.grouped_size;

    const bool accum_ok =
        scale == 1.0
            ? launch_vector_accumulate_kernel(
                  rhs_ctx.grouped_scratch.d_grouped_term,
                  rhs_ctx.grouped_scratch.d_grouped_accum,
                  grouped_elements,
                  stream)
            : launch_vector_scaled_accumulate_kernel(
                  rhs_ctx.grouped_scratch.d_grouped_term,
                  scale,
                  rhs_ctx.grouped_scratch.d_grouped_accum,
                  grouped_elements,
                  stream);

    return accum_ok ? 0 : PETSC_ERR_LIB;
}

// Closed-system per-layer left action: regroup |psi>, assemble the effective Hamiltonian, and
// accumulate -i*H_eff|psi> straight into f via the structured diagonal / sparse LEFT kernels.
// The one-sided, dissipator-free counterpart of apply_grouped_layout_terms_to_rhs.
PetscErrorCode apply_grouped_layout_terms_to_psi_rhs(
    const Solver& solver,
    PetscCudaTsRhsContext& rhs_ctx,
    Vec x,
    Vec f,
    CachedGroupedLayoutEntry& layout_entry,
    PetscReal t)
{
    cudaStream_t s = rhs_ctx.elementwise_stream;

    PetscCall(regroup_petsc_cuda_vec_to_psi_grouped_buffer(
        layout_entry.psi_cuda_grouped_layout,
        x,
        rhs_ctx.grouped_scratch.d_psi_grouped,
        rhs_ctx.batch_size,
        s));

    std::vector<Complex> effective_hamiltonian = layout_entry.static_hamiltonian_sum;
    bool has_effective_hamiltonian = !effective_hamiltonian.empty();
    bool has_dynamic_hamiltonian = false;
    for (const TimeDependentTerm& td_term : solver.model.time_dependent_hamiltonian_terms) {
        if (!same_sites(layout_entry.sites, td_term.sites)) {
            continue;
        }
        accumulate_operator_scaled(
            td_term.matrix,
            evaluate_time_dependent_coefficient(td_term, static_cast<double>(t)),
            effective_hamiltonian);
        has_effective_hamiltonian = true;
        has_dynamic_hamiltonian = true;
    }
    if (!has_effective_hamiltonian) {
        return 0;
    }

    const Index matrix_dim = static_cast<Index>(std::llround(std::sqrt(
        static_cast<double>(effective_hamiltonian.size()))));

    std::vector<Complex> diagonal;
    if (try_extract_local_diagonal(effective_hamiltonian, matrix_dim, diagonal)) {
        const void* d_diag = nullptr;
        if (has_dynamic_hamiltonian) {
            const std::size_t bytes = diagonal.size() * sizeof(Complex);
            if (layout_entry.dynamic_diagonal_capacity < diagonal.size()) {
                if (layout_entry.d_dynamic_diagonal != nullptr) {
                    cudaFree(layout_entry.d_dynamic_diagonal);
                    layout_entry.d_dynamic_diagonal = nullptr;
                }
                if (cudaMalloc(&layout_entry.d_dynamic_diagonal, bytes) != cudaSuccess) {
                    layout_entry.d_dynamic_diagonal = nullptr;
                    layout_entry.dynamic_diagonal_capacity = 0;
                    return PETSC_ERR_LIB;
                }
                layout_entry.dynamic_diagonal_capacity = diagonal.size();
            }
            if (cudaMemcpyAsync(layout_entry.d_dynamic_diagonal, diagonal.data(), bytes,
                                cudaMemcpyHostToDevice, s) != cudaSuccess) {
                return PETSC_ERR_LIB;
            }
            d_diag = layout_entry.d_dynamic_diagonal;
        } else {
            d_diag = layout_entry.d_static_hamiltonian_diagonal;
        }
        PetscCall(apply_grouped_diagonal_left_to_psi_vec(
            layout_entry.psi_cuda_grouped_layout, d_diag,
            rhs_ctx.grouped_scratch.d_psi_grouped, f, rhs_ctx.batch_size, s));
        return 0;
    }

    // Sparse 2-site (coupling) effective Hamiltonian -> sparse left action.
    if (has_dynamic_hamiltonian) {
        std::size_t dyn_nnz = 0;
        if (!upload_dynamic_sparse_hamiltonian(
                layout_entry, effective_hamiltonian, matrix_dim, dyn_nnz, s)) {
            return 0;  // all-zero effective H this step -> no contribution
        }
        // CSR sparse-left: visit only each output row's nonzeros (row_ptr built in the upload).
        PetscCall(apply_grouped_sparse_left_csr_to_psi_vec(
            layout_entry.psi_cuda_grouped_layout,
            layout_entry.d_dynamic_sparse_values,
            layout_entry.d_dynamic_sparse_cols,
            layout_entry.d_dynamic_sparse_row_ptr,
            rhs_ctx.grouped_scratch.d_psi_grouped, f, rhs_ctx.batch_size, s));
        return 0;
    }
    if (can_use_static_sparse_commutator(layout_entry, false)) {
        // Static (lab-frame) couplings: the all-nnz scan; no CSR row_ptr is cached for these.
        PetscCall(apply_grouped_sparse_left_to_psi_vec(
            layout_entry.psi_cuda_grouped_layout,
            layout_entry.d_static_sparse_hamiltonian_values,
            layout_entry.d_static_sparse_hamiltonian_rows,
            layout_entry.d_static_sparse_hamiltonian_cols,
            static_cast<Index>(layout_entry.static_sparse_hamiltonian_values.size()),
            rhs_ctx.grouped_scratch.d_psi_grouped, f, rhs_ctx.batch_size, s));
        return 0;
    }
    return PETSC_ERR_SUP;  // general cuTENSOR fallback not implemented for |psi> yet
}

// Fused single-site diagonal pre-pass: fold every single-site layer's effective diagonal into one
// full-Hilbert diagonal d_total[k] = sum_s h_s[occ_s(k)] (occ_s(k) = (k/full_stride[s])%dim[s]),
// then apply f += -i*d_total*|psi> in ONE regroup-free flat pass -- replacing N single-site
// regroups + N diagonal-left kernels. The per-layer loop must skip single-site layers when this
// runs. Single-site terms are diagonal by construction (omega*n + alpha*n(n-1) + drive*n).
PetscErrorCode apply_fused_psi_diagonal_prepass(
    const Solver& solver,
    PetscCudaTsRhsContext& rhs_ctx,
    Vec x,
    Vec f,
    PetscReal t)
{
    cudaStream_t s = rhs_ctx.elementwise_stream;
    const Index D = solver.layout.hilbert_dim;
    const std::vector<Index>& dims = solver.model.local_dims;
    const Index n = dims.size();
    std::vector<Index> full_stride(n, 1);
    for (Index i = n; i-- > 1;) {
        full_stride[i - 1] = full_stride[i] * dims[i];
    }

    // Gather only the tiny per-site diagonals + their [offset, full_stride, dim] descriptor; the
    // full-D diagonal is summed on the GPU (no host D-loop).
    std::vector<Complex> site_diags;
    std::vector<Index> site_desc;
    for (CachedGroupedLayoutEntry& entry : rhs_ctx.cached_grouped_layouts) {
        if (entry.sites.size() != 1) {
            continue;
        }
        std::vector<Complex> effective_hamiltonian = entry.static_hamiltonian_sum;
        bool has_effective_hamiltonian = !effective_hamiltonian.empty();
        for (const TimeDependentTerm& td_term : solver.model.time_dependent_hamiltonian_terms) {
            if (!same_sites(entry.sites, td_term.sites)) {
                continue;
            }
            accumulate_operator_scaled(
                td_term.matrix,
                evaluate_time_dependent_coefficient(td_term, static_cast<double>(t)),
                effective_hamiltonian);
            has_effective_hamiltonian = true;
        }
        if (!has_effective_hamiltonian) {
            continue;
        }
        const Index matrix_dim = static_cast<Index>(std::llround(std::sqrt(
            static_cast<double>(effective_hamiltonian.size()))));
        std::vector<Complex> diagonal;
        if (!try_extract_local_diagonal(effective_hamiltonian, matrix_dim, diagonal)) {
            continue;  // single-site is diagonal by construction; defensive skip
        }
        const Index site = entry.sites[0];
        site_desc.push_back(static_cast<Index>(site_diags.size()));
        site_desc.push_back(full_stride[site]);
        site_desc.push_back(dims[site]);
        site_diags.insert(site_diags.end(), diagonal.begin(), diagonal.end());
    }
    const Index num_folded = static_cast<Index>(site_desc.size() / 3);
    if (num_folded == 0) {
        return 0;
    }
    if (cudaMemcpyAsync(rhs_ctx.grouped_scratch.d_psi_site_diags, site_diags.data(),
                        site_diags.size() * sizeof(Complex),
                        cudaMemcpyHostToDevice, s) != cudaSuccess ||
        cudaMemcpyAsync(rhs_ctx.grouped_scratch.d_psi_site_desc, site_desc.data(),
                        site_desc.size() * sizeof(Index),
                        cudaMemcpyHostToDevice, s) != cudaSuccess) {
        return PETSC_ERR_LIB;
    }
    PetscCall(apply_psi_fused_diagonal_to_petsc_vec(
        rhs_ctx.grouped_scratch.d_psi_site_diags,
        rhs_ctx.grouped_scratch.d_psi_site_desc,
        num_folded, x, f, D, rhs_ctx.batch_size, s));
    return 0;
}

PetscErrorCode apply_grouped_layout_terms_to_rhs(
    const Solver& solver,
    PetscCudaTsRhsContext& rhs_ctx,
    Vec x,
    Vec f,
    CachedGroupedLayoutEntry& layout_entry,
    PetscReal t)
{
    cudaStream_t s = rhs_ctx.elementwise_stream;
    if (layout_entry.grouped_bytes > rhs_ctx.grouped_scratch.grouped_bytes) {
        return PETSC_ERR_ARG_SIZ;
    }

    PetscCall(regroup_petsc_cuda_vec_to_grouped_buffer(
        layout_entry.cuda_grouped_layout,
        x,
        rhs_ctx.grouped_scratch.d_grouped_input,
        rhs_ctx.batch_size,
        s));

    bool has_contribution = false;
    std::vector<Complex> effective_hamiltonian =
        layout_entry.static_hamiltonian_sum;
    bool has_effective_hamiltonian = !effective_hamiltonian.empty();
    bool has_dynamic_hamiltonian = false;

    for (const TimeDependentTerm& td_term : solver.model.time_dependent_hamiltonian_terms) {
        if (!same_sites(layout_entry.sites, td_term.sites)) {
            continue;
        }

        accumulate_operator_scaled(
            td_term.matrix,
            evaluate_time_dependent_coefficient(td_term, static_cast<double>(t)),
            effective_hamiltonian);
        has_effective_hamiltonian = true;
        has_dynamic_hamiltonian = true;
    }

    // ---- Fused single-site diagonal fast path (opt-in via -tc_fused_diagonal) ----
    // For layers whose effective Hamiltonian is diagonal (single-site bare terms;
    // couplings are sparse and fall through), combine the commutator, the anti-commutator,
    // and one diagonal dissipator jump into a single OVERWRITING pass, then apply the
    // remaining (relaxation dense / extra diagonal) jumps. Saves ~2 full-state passes per
    // single-site layer. Not bit-identical (float summation order); validated vs the
    // separate path (run with -tc_fused_diagonal 0) and QuTiP.
    static int s_fused_diagonal = -1;
    if (s_fused_diagonal < 0) {
        PetscInt fused_opt = 0;
        PetscOptionsGetInt(nullptr, nullptr, "-tc_fused_diagonal", &fused_opt, nullptr);
        s_fused_diagonal = (fused_opt != 0) ? 1 : 0;
    }
    // -tc_fuse_coupling also routes the single-site fused-diagonal result straight into f
    // (skipping the grouped accumulator + flatten), in addition to the coupling fusion below.
    static int s_fuse_coupling = -1;
    if (s_fuse_coupling < 0) {
        PetscInt fuse_opt = 0;
        PetscOptionsGetInt(nullptr, nullptr, "-tc_fuse_coupling", &fuse_opt, nullptr);
        s_fuse_coupling = (fuse_opt != 0) ? 1 : 0;
    }
    if (s_fused_diagonal == 1 && has_effective_hamiltonian) {
        const Index matrix_dim = static_cast<Index>(std::llround(std::sqrt(
            static_cast<double>(effective_hamiltonian.size()))));
        std::vector<Complex> h_diagonal;
        bool fusable =
            matrix_dim > 0 &&
            layout_entry.grouped_layout.hilbert_dim % matrix_dim == 0 &&
            try_extract_local_diagonal(effective_hamiltonian, matrix_dim, h_diagonal);
        // A present dissipator-norm must be diagonal, and every dissipator on the layer must
        // be structured (diagonal jump or dense single-site) -- else fall back to the
        // separate path (which can use the cuTENSOR contraction).
        if (fusable && !layout_entry.static_dissipator_norm_sum.empty() &&
            (layout_entry.static_dissipator_norm_diagonal.empty() ||
             layout_entry.d_static_dissipator_norm_diagonal == nullptr)) {
            fusable = false;
        }
        if (fusable) {
            for (const CachedDissipatorAuxiliaries& d_aux : rhs_ctx.cached_static_dissipators) {
                if (same_sites(layout_entry.sites, d_aux.sites) &&
                    d_aux.jump_diagonal.empty() && d_aux.d_l_op == nullptr) {
                    fusable = false;
                    break;
                }
            }
        }
        if (fusable) {
            const Index complement_dim =
                layout_entry.grouped_layout.hilbert_dim / matrix_dim;
            // Upload the effective Hamiltonian diagonal (static or dynamic) -> d_dynamic_diagonal.
            const std::size_t bytes = h_diagonal.size() * sizeof(Complex);
            if (layout_entry.dynamic_diagonal_capacity < h_diagonal.size()) {
                if (layout_entry.d_dynamic_diagonal != nullptr) {
                    cudaFree(layout_entry.d_dynamic_diagonal);
                    layout_entry.d_dynamic_diagonal = nullptr;
                }
                if (cudaMalloc(&layout_entry.d_dynamic_diagonal, bytes) != cudaSuccess) {
                    layout_entry.d_dynamic_diagonal = nullptr;
                    layout_entry.dynamic_diagonal_capacity = 0;
                    return PETSC_ERR_LIB;
                }
                layout_entry.dynamic_diagonal_capacity = h_diagonal.size();
            }
            if (cudaMemcpyAsync(layout_entry.d_dynamic_diagonal, h_diagonal.data(), bytes,
                                cudaMemcpyHostToDevice, s) != cudaSuccess) {
                return PETSC_ERR_LIB;
            }
            // First diagonal-jump dissipator is folded into the fused kernel.
            const void* fused_jump = nullptr;
            for (const CachedDissipatorAuxiliaries& d_aux : rhs_ctx.cached_static_dissipators) {
                if (same_sites(layout_entry.sites, d_aux.sites) && !d_aux.jump_diagonal.empty()) {
                    fused_jump = d_aux.d_jump_diagonal;
                    break;
                }
            }
            const ScopedNvtxRange fused_range(make_nvtx_label(
                "rhs_fused_diagonal", make_sites_cache_suffix(layout_entry.sites)));
            const void* norm_ptr =
                layout_entry.static_dissipator_norm_sum.empty()
                    ? nullptr
                    : layout_entry.d_static_dissipator_norm_diagonal;
            int n_diag_jumps = 0;
            for (const CachedDissipatorAuxiliaries& d_aux : rhs_ctx.cached_static_dissipators) {
                if (same_sites(layout_entry.sites, d_aux.sites) && !d_aux.jump_diagonal.empty()) {
                    ++n_diag_jumps;
                }
            }
            // Fuse-to-flat: write the single-site contribution (fused diagonal + relax dense
            // jumps) straight into f, skipping the grouped accumulator + the flatten. Only
            // when there is no extra diagonal jump (the fused kernel folds in exactly one;
            // there is no diagonal-jump-to-flat variant).
            if (s_fuse_coupling == 1 && n_diag_jumps <= 1) {
                PetscCall(apply_grouped_fused_diagonal_to_petsc_vec(
                    layout_entry.cuda_grouped_layout,
                    layout_entry.d_dynamic_diagonal, norm_ptr, fused_jump,
                    rhs_ctx.grouped_scratch.d_grouped_input,
                    f, rhs_ctx.batch_size, s));
                for (const CachedDissipatorAuxiliaries& d_aux : rhs_ctx.cached_static_dissipators) {
                    if (same_sites(layout_entry.sites, d_aux.sites) &&
                        d_aux.jump_diagonal.empty() && d_aux.d_l_op != nullptr) {
                        PetscCall(apply_grouped_dense_jump_to_petsc_vec(
                            layout_entry.cuda_grouped_layout, d_aux.d_l_op,
                            rhs_ctx.grouped_scratch.d_grouped_input,
                            f, rhs_ctx.batch_size, s));
                    }
                }
                return 0;
            }
            // Default: accumulate into the grouped buffer, then flatten into f.
            PetscCall(ensure_grouped_accum_buffer(rhs_ctx.grouped_scratch));
            if (!launch_batched_grouped_fused_diagonal_kernel(
                    layout_entry.d_dynamic_diagonal,
                    norm_ptr,
                    fused_jump,
                    rhs_ctx.grouped_scratch.d_grouped_input,
                    rhs_ctx.grouped_scratch.d_grouped_accum,
                    matrix_dim, complement_dim, rhs_ctx.batch_size, s)) {
                return PETSC_ERR_LIB;
            }
            // Apply remaining dissipators: relaxation dense jumps + any extra diagonal jumps.
            bool fused_jump_seen = false;
            for (const CachedDissipatorAuxiliaries& d_aux : rhs_ctx.cached_static_dissipators) {
                if (!same_sites(layout_entry.sites, d_aux.sites)) {
                    continue;
                }
                if (!d_aux.jump_diagonal.empty()) {
                    if (!fused_jump_seen && d_aux.d_jump_diagonal == fused_jump) {
                        fused_jump_seen = true;
                        continue;  // already folded into the fused kernel
                    }
                    const Index tdim = static_cast<Index>(d_aux.jump_diagonal.size());
                    const Index cdim = layout_entry.grouped_layout.hilbert_dim / tdim;
                    if (!launch_batched_grouped_diagonal_dissipator_jump_kernel(
                            d_aux.d_jump_diagonal,
                            rhs_ctx.grouped_scratch.d_grouped_input,
                            rhs_ctx.grouped_scratch.d_grouped_accum,
                            tdim, cdim, rhs_ctx.batch_size, s)) {
                        return PETSC_ERR_LIB;
                    }
                } else if (d_aux.d_l_op != nullptr) {
                    const Index tdim = static_cast<Index>(std::llround(std::sqrt(
                        static_cast<double>(d_aux.l_op.size()))));
                    const Index cdim = layout_entry.grouped_layout.hilbert_dim / tdim;
                    if (!launch_batched_grouped_dense_jump_kernel(
                            d_aux.d_l_op,
                            rhs_ctx.grouped_scratch.d_grouped_input,
                            rhs_ctx.grouped_scratch.d_grouped_accum,
                            tdim, cdim, rhs_ctx.batch_size, s)) {
                        return PETSC_ERR_LIB;
                    }
                }
            }
            PetscCall(flatten_grouped_buffer_accumulate_into_petsc_cuda_vec(
                layout_entry.cuda_grouped_layout,
                rhs_ctx.grouped_scratch.d_grouped_accum,
                f, rhs_ctx.batch_size, s));
            return 0;
        }
    }

    // ---- Fused coupling sparse-commutator-to-flat fast path (opt-in via -tc_fuse_coupling) ----
    // A pure 2-site coupling layer has the sparse commutator as its SOLE contributor (no
    // dissipator). Apply -i[H,rho] straight into f, skipping the grouped accumulator write +
    // the separate flatten pass (removes one full grouped-buffer round-trip per coupling).
    if (s_fuse_coupling == 1 && has_effective_hamiltonian &&
        layout_entry.sites.size() == 2 &&
        layout_entry.static_dissipator_norm_sum.empty()) {
        bool layer_has_dissipator = false;
        for (const CachedDissipatorAuxiliaries& d_aux : rhs_ctx.cached_static_dissipators) {
            if (same_sites(layout_entry.sites, d_aux.sites)) {
                layer_has_dissipator = true;
                break;
            }
        }
        if (!layer_has_dissipator) {
            const Index matrix_dim = static_cast<Index>(std::llround(std::sqrt(
                static_cast<double>(effective_hamiltonian.size()))));
            const void* sv = nullptr;
            const void* sr = nullptr;
            const void* sc = nullptr;
            Index nnz = 0;
            bool ready = false;
            if (has_dynamic_hamiltonian) {
                std::size_t dyn_nnz = 0;
                if (upload_dynamic_sparse_hamiltonian(
                        layout_entry, effective_hamiltonian, matrix_dim, dyn_nnz, s)) {
                    sv = layout_entry.d_dynamic_sparse_values;
                    sr = layout_entry.d_dynamic_sparse_rows;
                    sc = layout_entry.d_dynamic_sparse_cols;
                    nnz = static_cast<Index>(dyn_nnz);
                    ready = true;
                }
            } else if (can_use_static_sparse_commutator(layout_entry, false)) {
                sv = layout_entry.d_static_sparse_hamiltonian_values;
                sr = layout_entry.d_static_sparse_hamiltonian_rows;
                sc = layout_entry.d_static_sparse_hamiltonian_cols;
                nnz = static_cast<Index>(layout_entry.static_sparse_hamiltonian_values.size());
                ready = true;
            }
            if (ready) {
                const ScopedNvtxRange fused_coupling_range(make_nvtx_label(
                    "rhs_fused_coupling", make_sites_cache_suffix(layout_entry.sites)));
                PetscCall(apply_grouped_sparse_commutator_to_petsc_vec(
                    layout_entry.cuda_grouped_layout,
                    sv, sr, sc, nnz,
                    rhs_ctx.grouped_scratch.d_grouped_input,
                    f, rhs_ctx.batch_size, s));
                return 0;
            }
        }
    }

    // Fall-through (non-fused / cuTENSOR) path: this layer accumulates into the grouped buffer
    // before a flatten, so the accumulator must exist (lazily allocated; never touched on the
    // fully-fused production path, where it stays unallocated to save ~0.64 GiB/state).
    PetscCall(ensure_grouped_accum_buffer(rhs_ctx.grouped_scratch));

    // The commutator (first term, applied by an overwriting fast-path kernel) initializes
    // the accumulator for this layer, so the explicit per-layer zeroing is only needed when
    // no commutator is present (e.g. a dissipator-only layer). This removes a full-buffer
    // memset per layer per RHS (~5% of GPU time at 4Q4C). The cuTENSOR commutator fallback
    // re-zeros itself below.
    if (!has_effective_hamiltonian) {
        PetscCall(zero_grouped_layout_accumulator(
            rhs_ctx, layout_entry, rhs_ctx.batch_size, s));
    }

    if (has_effective_hamiltonian) {
        const std::string effective_term_label =
            has_dynamic_hamiltonian
                ? "layout_effective_dynamic_" +
                      make_sites_cache_suffix(layout_entry.sites) +
                      "_t_" + std::to_string(static_cast<double>(t))
                : "layout_effective_static_" +
                      make_sites_cache_suffix(layout_entry.sites);

        const ScopedNvtxRange commutator_range(
            make_nvtx_label("rhs_commutator", effective_term_label));
        if (can_use_static_sparse_commutator(layout_entry, has_dynamic_hamiltonian)) {
            const Index target_hilbert_dim =
                static_cast<Index>(std::llround(std::sqrt(
                    static_cast<double>(layout_entry.static_hamiltonian_sum.size()))));
            const Index complement_dim =
                layout_entry.grouped_layout.hilbert_dim / target_hilbert_dim;
            if (!launch_batched_grouped_sparse_commutator_kernel(
                    layout_entry.d_static_sparse_hamiltonian_values,
                    layout_entry.d_static_sparse_hamiltonian_rows,
                    layout_entry.d_static_sparse_hamiltonian_cols,
                    static_cast<Index>(layout_entry.static_sparse_hamiltonian_values.size()),
                    rhs_ctx.grouped_scratch.d_grouped_input,
                    rhs_ctx.grouped_scratch.d_grouped_accum,
                    target_hilbert_dim,
                    complement_dim,
                    rhs_ctx.batch_size,
                    s)) {
                return PETSC_ERR_LIB;
            }
        } else if (can_use_static_diagonal_commutator(layout_entry, has_dynamic_hamiltonian)) {
            const Index target_hilbert_dim =
                static_cast<Index>(layout_entry.static_hamiltonian_diagonal.size());
            const Index complement_dim =
                layout_entry.grouped_layout.hilbert_dim / target_hilbert_dim;
            if (!launch_batched_grouped_diagonal_commutator_kernel(
                    layout_entry.d_static_hamiltonian_diagonal,
                    rhs_ctx.grouped_scratch.d_grouped_input,
                    rhs_ctx.grouped_scratch.d_grouped_accum,
                    target_hilbert_dim,
                    complement_dim,
                    rhs_ctx.batch_size,
                    s)) {
                return PETSC_ERR_LIB;
            }
        } else if (has_dynamic_hamiltonian &&
                   try_apply_dynamic_structured_commutator(
                       layout_entry,
                       effective_hamiltonian,
                       rhs_ctx.grouped_scratch.d_grouped_input,
                       rhs_ctx.grouped_scratch.d_grouped_accum,
                       rhs_ctx.batch_size,
                       s)) {
            // Dynamic (time-dependent) effective Hamiltonian applied via a structured
            // fast path (diagonal / sparse), accumulated directly into d_grouped_accum.
        } else {
            // General cuTENSOR fallback writes the term buffer, then accumulates. Unlike
            // the overwriting fast-path kernels, it does not initialize the accumulator, so
            // (since the per-layer zero is now skipped when a commutator is present) zero it
            // here first. Dead for the structured tunable-coupler model; kept correct.
            PetscCall(zero_grouped_layout_accumulator(
                rhs_ctx, layout_entry, rhs_ctx.batch_size, s));
            PetscCall(ensure_grouped_term_buffer(rhs_ctx.grouped_scratch));
            PetscCall(apply_grouped_commutator_cuda_buffer(
                solver,
                effective_term_label,
                effective_hamiltonian,
                layout_entry.sites,
                layout_entry.grouped_layout,
                rhs_ctx.executor_cache,
                rhs_ctx.grouped_scratch.d_grouped_input,
                rhs_ctx.grouped_scratch.d_grouped_term,
                rhs_ctx.batch_size,
                s,
                nullptr,
                nullptr));
            PetscCall(accumulate_grouped_layout_term(
                rhs_ctx, layout_entry, 1.0, rhs_ctx.batch_size, s));
        }

        has_contribution = true;
    }

    for (const CachedDissipatorAuxiliaries& d_aux : rhs_ctx.cached_static_dissipators) {
        if (!same_sites(layout_entry.sites, d_aux.sites)) {
            continue;
        }

        if (!d_aux.jump_diagonal.empty()) {
            const ScopedNvtxRange diagonal_jump_range(
                make_nvtx_label("rhs_dissipator_diag_jump", d_aux.name));
            const Index target_hilbert_dim =
                static_cast<Index>(d_aux.jump_diagonal.size());
            const Index complement_dim =
                layout_entry.grouped_layout.hilbert_dim / target_hilbert_dim;

            if (!launch_batched_grouped_diagonal_dissipator_jump_kernel(
                    d_aux.d_jump_diagonal,
                    rhs_ctx.grouped_scratch.d_grouped_input,
                    rhs_ctx.grouped_scratch.d_grouped_accum,
                    target_hilbert_dim,
                    complement_dim,
                    rhs_ctx.batch_size,
                    s)) {
                return PETSC_ERR_LIB;
            }
        } else if (d_aux.d_l_op != nullptr) {
            // Non-diagonal single-site collapse op (e.g. relaxation a): dense jump
            // kernel L rho L^dag instead of a general cuTENSOR contraction.
            const Index target_hilbert_dim =
                static_cast<Index>(std::llround(std::sqrt(
                    static_cast<double>(d_aux.l_op.size()))));
            const Index complement_dim =
                layout_entry.grouped_layout.hilbert_dim / target_hilbert_dim;
            const ScopedNvtxRange jump_range(
                make_nvtx_label("rhs_dissipator_jump_dense", d_aux.name));
            if (!launch_batched_grouped_dense_jump_kernel(
                    d_aux.d_l_op,
                    rhs_ctx.grouped_scratch.d_grouped_input,
                    rhs_ctx.grouped_scratch.d_grouped_accum,
                    target_hilbert_dim,
                    complement_dim,
                    rhs_ctx.batch_size,
                    s)) {
                return PETSC_ERR_LIB;
            }
        } else {
            const ScopedNvtxRange jump_range(
                make_nvtx_label("rhs_dissipator_jump", d_aux.name));
            PetscCall(ensure_grouped_term_buffer(rhs_ctx.grouped_scratch));
            PetscCall(apply_grouped_dissipator_jump_cuda_buffer(
                solver,
                d_aux.name,
                d_aux.l_op,
                d_aux.l_dag,
                d_aux.sites,
                layout_entry.grouped_layout,
                rhs_ctx.executor_cache,
                rhs_ctx.grouped_scratch.d_grouped_input,
                rhs_ctx.grouped_scratch.d_grouped_term,
                rhs_ctx.batch_size,
                s,
                nullptr,
                nullptr));
            PetscCall(accumulate_grouped_layout_term(
                rhs_ctx, layout_entry, 1.0, rhs_ctx.batch_size, s));
        }

        has_contribution = true;
    }

    if (!layout_entry.static_dissipator_norm_sum.empty()) {
        const std::string dissipator_norm_label =
            "layout_diss_norm_" + make_sites_cache_suffix(layout_entry.sites);

        const ScopedNvtxRange anti_commutator_range(
            make_nvtx_label("rhs_anti_commutator", dissipator_norm_label));
        if (!layout_entry.static_dissipator_norm_diagonal.empty() &&
            layout_entry.d_static_dissipator_norm_diagonal != nullptr) {
            // Diagonal norm sum (sum_k L_k^dag L_k) -> cheap diagonal anti-commutator
            // kernel instead of a general cuTENSOR contraction.
            const Index target_hilbert_dim =
                static_cast<Index>(layout_entry.static_dissipator_norm_diagonal.size());
            const Index complement_dim =
                layout_entry.grouped_layout.hilbert_dim / target_hilbert_dim;
            if (!launch_batched_grouped_diagonal_anti_commutator_kernel(
                    layout_entry.d_static_dissipator_norm_diagonal,
                    rhs_ctx.grouped_scratch.d_grouped_input,
                    rhs_ctx.grouped_scratch.d_grouped_accum,
                    target_hilbert_dim,
                    complement_dim,
                    rhs_ctx.batch_size,
                    s)) {
                return PETSC_ERR_LIB;
            }
        } else {
            PetscCall(ensure_grouped_term_buffer(rhs_ctx.grouped_scratch));
            PetscCall(apply_grouped_anti_commutator_cuda_buffer(
                solver,
                dissipator_norm_label,
                layout_entry.static_dissipator_norm_sum,
                layout_entry.sites,
                layout_entry.grouped_layout,
                rhs_ctx.executor_cache,
                rhs_ctx.grouped_scratch.d_grouped_input,
                rhs_ctx.grouped_scratch.d_grouped_term,
                rhs_ctx.batch_size,
                s,
                nullptr,
                nullptr));
            PetscCall(accumulate_grouped_layout_term(
                rhs_ctx, layout_entry, 1.0, rhs_ctx.batch_size, s));
        }

        has_contribution = true;
    }

    if (!has_contribution) {
        return 0;
    }

    // Accumulate this layer's flattened contribution straight into f (zeroed once per
    // RHS step), avoiding a per-layer temporary vector + add + copy.
    PetscCall(flatten_grouped_buffer_accumulate_into_petsc_cuda_vec(
        layout_entry.cuda_grouped_layout,
        rhs_ctx.grouped_scratch.d_grouped_accum,
        f,
        rhs_ctx.batch_size,
        s));

    return 0;
}

} // namespace

PetscErrorCode ts_rhs_function_cuda_grouped_commutator(
    TS,
    PetscReal,
    Vec x,
    Vec f,
    void* ctx)
{
    auto* rhs_ctx = static_cast<PetscCudaTsRhsContext*>(ctx);
    if (!rhs_ctx || !rhs_ctx->solver || !rhs_ctx->h_local_op) {
        return PETSC_ERR_ARG_NULL;
    }

    PetscCall(apply_grouped_commutator_cuda_vec(
        *rhs_ctx->solver,
        "smoke_h",
        *rhs_ctx->h_local_op,
        rhs_ctx->target_sites,
        rhs_ctx->grouped_layout,
        rhs_ctx->cuda_grouped_layout,
        rhs_ctx->executor_cache,
        x,
        f,
        rhs_ctx->batch_size,
        rhs_ctx->elementwise_stream));

    if (cudaStreamSynchronize(rhs_ctx->elementwise_stream) != cudaSuccess) {
        return PETSC_ERR_LIB;
    }

    return 0;
}

PetscErrorCode ts_rhs_function_cuda_grouped_liouvillian(
    TS,
    PetscReal,
    Vec x,
    Vec f,
    void* ctx)
{
    auto* rhs_ctx = static_cast<PetscCudaTsRhsContext*>(ctx);
    if (!rhs_ctx || !rhs_ctx->solver || !rhs_ctx->h_local_op ||
        !rhs_ctx->d_local_op || !rhs_ctx->d_local_op_dag || !rhs_ctx->d_local_op_dag_op) {
        return PETSC_ERR_ARG_NULL;
    }

    cudaStream_t s = rhs_ctx->elementwise_stream;

    Vec temp_comm = rhs_ctx->work_vec_a;
    Vec temp_diss = rhs_ctx->work_vec_b;

    PetscCall(apply_grouped_commutator_cuda_vec(
        *rhs_ctx->solver,
        "smoke_h",
        *rhs_ctx->h_local_op,
        rhs_ctx->target_sites,
        rhs_ctx->grouped_layout,
        rhs_ctx->cuda_grouped_layout,
        rhs_ctx->executor_cache,
        x,
        temp_comm,
        rhs_ctx->batch_size,
        s));

    PetscCall(apply_grouped_dissipator_cuda_vec(
        *rhs_ctx->solver,
        "smoke_d",
        *rhs_ctx->d_local_op,
        *rhs_ctx->d_local_op_dag,
        *rhs_ctx->d_local_op_dag_op,
        rhs_ctx->target_sites,
        rhs_ctx->grouped_layout,
        rhs_ctx->cuda_grouped_layout,
        rhs_ctx->executor_cache,
        x,
        temp_diss,
        rhs_ctx->batch_size,
        s));

    PetscCall(add_petsc_cuda_vecs(
        temp_comm,
        temp_diss,
        f,
        rhs_total_density_dim(*rhs_ctx),
        s));

    if (cudaStreamSynchronize(s) != cudaSuccess) {
        return PETSC_ERR_LIB;
    }

    return 0;
}

PetscErrorCode ts_rhs_function_cuda_static_model_liouvillian(
    TS,
    PetscReal,
    Vec x,
    Vec f,
    void* ctx)
{
    auto* rhs_ctx = static_cast<PetscCudaTsRhsContext*>(ctx);
    if (!rhs_ctx || !rhs_ctx->solver) {
        return PETSC_ERR_ARG_NULL;
    }

    const Solver& solver = *rhs_ctx->solver;
    cudaStream_t s = rhs_ctx->elementwise_stream;

    const Index total_density_dim = rhs_total_density_dim(*rhs_ctx);
    PetscCall(zero_petsc_cuda_vec(f, total_density_dim, s));

    for (CachedGroupedLayoutEntry& layout_entry : rhs_ctx->cached_grouped_layouts) {
        PetscCall(apply_grouped_layout_terms_to_rhs(
            solver,
            *rhs_ctx,
            x,
            f,
            layout_entry,
            0.0));
    }

    if (cudaStreamSynchronize(s) != cudaSuccess) {
        return PETSC_ERR_LIB;
    }

    return 0;
}

PetscErrorCode ts_rhs_function_cuda_full_model_liouvillian(
    TS,
    PetscReal t,
    Vec x,
    Vec f,
    void* ctx)
{
    auto* rhs_ctx = static_cast<PetscCudaTsRhsContext*>(ctx);
    if (!rhs_ctx || !rhs_ctx->solver) {
        return PETSC_ERR_ARG_NULL;
    }

    const Solver& solver = *rhs_ctx->solver;
    cudaStream_t s = rhs_ctx->elementwise_stream;

    const Index total_density_dim = rhs_total_density_dim(*rhs_ctx);
    PetscCall(zero_petsc_cuda_vec(f, total_density_dim, s));

    for (CachedGroupedLayoutEntry& layout_entry : rhs_ctx->cached_grouped_layouts) {
        PetscCall(apply_grouped_layout_terms_to_rhs(
            solver,
            *rhs_ctx,
            x,
            f,
            layout_entry,
            t));
    }

    if (cudaStreamSynchronize(s) != cudaSuccess) {
        return PETSC_ERR_LIB;
    }

    return 0;
}

PetscErrorCode ts_rhs_function_cuda_full_model_schrodinger(
    TS,
    PetscReal t,
    Vec x,
    Vec f,
    void* ctx)
{
    auto* rhs_ctx = static_cast<PetscCudaTsRhsContext*>(ctx);
    if (!rhs_ctx || !rhs_ctx->solver) {
        return PETSC_ERR_ARG_NULL;
    }

    const Solver& solver = *rhs_ctx->solver;
    cudaStream_t s = rhs_ctx->elementwise_stream;

    const Index total_hilbert_dim = rhs_total_hilbert_dim(*rhs_ctx);
    PetscCall(zero_petsc_cuda_vec(f, total_hilbert_dim, s));

    // Fused single-site diagonal pre-pass (opt-in): one flat multiply replaces N single-site
    // regroups + diagonal kernels. The per-layer loop then skips single-site layers.
    static int s_fused_psi_diagonal = -1;
    if (s_fused_psi_diagonal < 0) {
        PetscInt fused_opt = 0;
        PetscOptionsGetInt(nullptr, nullptr, "-tc_fused_psi_diagonal", &fused_opt, nullptr);
        s_fused_psi_diagonal = (fused_opt != 0) ? 1 : 0;
    }
    if (s_fused_psi_diagonal == 1) {
        PetscCall(apply_fused_psi_diagonal_prepass(solver, *rhs_ctx, x, f, t));
    }

    for (CachedGroupedLayoutEntry& layout_entry : rhs_ctx->cached_grouped_layouts) {
        if (s_fused_psi_diagonal == 1 && layout_entry.sites.size() == 1) {
            continue;  // handled by the fused diagonal pre-pass
        }
        PetscCall(apply_grouped_layout_terms_to_psi_rhs(
            solver, *rhs_ctx, x, f, layout_entry, t));
    }

    if (cudaStreamSynchronize(s) != cudaSuccess) {
        return PETSC_ERR_LIB;
    }

    return 0;
}

} // namespace culindblad

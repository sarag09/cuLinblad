#include "culindblad/batch_evolution.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include <petscts.h>
#include <petscvec.h>

#include <cuda_runtime.h>

#include "culindblad/backend.hpp"
#include "culindblad/cuda_elementwise.hpp"
#include "culindblad/cuda_grouped_layout.hpp"
#include "culindblad/cutensor_contraction_desc.hpp"
#include "culindblad/cutensor_executor.hpp"
#include "culindblad/cutensor_executor_cache.hpp"
#include "culindblad/grouped_state_layout.hpp"
#include "culindblad/k_site_operator_embed.hpp"
#include "culindblad/local_operator_utils.hpp"
#include "culindblad/time_dependent_term.hpp"

namespace culindblad {

namespace {

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

bool should_use_tiny_dense_path(
    const Solver& solver,
    const std::vector<Index>& target_sites)
{
    return target_sites.size() <= 2 &&
           solver.layout.hilbert_dim <= kMaxTinyDenseHilbertDim;
}

void build_tiny_dense_operator_arg(
    const std::vector<Complex>& full_op,
    Index dim,
    TinyDenseOperatorKernelArg& arg)
{
    if (dim > kMaxTinyDenseHilbertDim) {
        throw std::invalid_argument(
            "build_tiny_dense_operator_arg: dimension exceeds kernel limit");
    }

    if (full_op.size() != dim * dim) {
        throw std::invalid_argument(
            "build_tiny_dense_operator_arg: operator size mismatch");
    }

    arg.dim = dim;
    for (Index i = 0; i < kMaxTinyDenseOperatorElements; ++i) {
        arg.data[i] = Complex{0.0, 0.0};
    }

    for (Index i = 0; i < full_op.size(); ++i) {
        arg.data[i] = full_op[i];
    }
}

std::vector<Index> choose_seed_target_sites(
    const Solver& solver)
{
    if (!solver.model.hamiltonian_terms.empty() &&
        !solver.model.hamiltonian_terms.front().sites.empty()) {
        return solver.model.hamiltonian_terms.front().sites;
    }

    if (!solver.model.dissipator_terms.empty() &&
        !solver.model.dissipator_terms.front().sites.empty()) {
        return solver.model.dissipator_terms.front().sites;
    }

    if (!solver.model.time_dependent_hamiltonian_terms.empty() &&
        !solver.model.time_dependent_hamiltonian_terms.front().sites.empty()) {
        return solver.model.time_dependent_hamiltonian_terms.front().sites;
    }

    if (solver.model.local_dims.empty()) {
        throw std::runtime_error(
            "choose_seed_target_sites: solver model has no subsystems");
    }

    return {0};
}

std::vector<CachedDissipatorAuxiliaries> build_cached_static_dissipators(
    const Solver& solver)
{
    std::vector<CachedDissipatorAuxiliaries> cached;

    for (const OperatorTerm& d_term : solver.model.dissipator_terms) {
        CachedDissipatorAuxiliaries aux;
        aux.name = d_term.name;
        aux.sites = d_term.sites;
        aux.l_op = d_term.matrix;
        aux.l_dag = local_conjugate_transpose(d_term.matrix, d_term.row_dim);
        aux.l_dag_l = local_multiply_square(aux.l_dag, d_term.matrix, d_term.row_dim);
        cached.push_back(std::move(aux));
    }

    return cached;
}

std::vector<CachedGroupedLayoutEntry> build_cached_grouped_layouts(
    const Solver& solver)
{
    std::vector<CachedGroupedLayoutEntry> cached;

    auto add_if_missing = [&](const std::vector<Index>& sites) {
        for (const CachedGroupedLayoutEntry& entry : cached) {
            if (same_sites(entry.sites, sites)) {
                return;
            }
        }

        CachedGroupedLayoutEntry entry;
        entry.sites = sites;
        entry.grouped_layout =
            make_grouped_state_layout(sites, solver.model.local_dims);

        if (!create_cuda_grouped_state_layout(
                entry.grouped_layout,
                entry.cuda_grouped_layout)) {
            throw std::runtime_error(
                "build_cached_grouped_layouts: failed to create CUDA grouped layout");
        }

        cached.push_back(std::move(entry));
    };

    for (const OperatorTerm& h_term : solver.model.hamiltonian_terms) {
        add_if_missing(h_term.sites);
    }

    for (const OperatorTerm& d_term : solver.model.dissipator_terms) {
        add_if_missing(d_term.sites);
    }

    for (const TimeDependentTerm& td_term : solver.model.time_dependent_hamiltonian_terms) {
        add_if_missing(td_term.sites);
    }

    return cached;
}

void destroy_cached_grouped_layouts(
    std::vector<CachedGroupedLayoutEntry>& cached)
{
    for (CachedGroupedLayoutEntry& entry : cached) {
        (void)destroy_cuda_grouped_state_layout(entry.cuda_grouped_layout);
    }
    cached.clear();
}

PetscErrorCode create_rhs_work_vectors(
    Vec prototype,
    PetscCudaTsRhsContext& rhs_ctx)
{
    rhs_ctx.work_vec_a = nullptr;
    rhs_ctx.work_vec_b = nullptr;
    rhs_ctx.work_vec_c = nullptr;

    PetscCall(VecDuplicate(prototype, &rhs_ctx.work_vec_a));
    PetscCall(VecDuplicate(prototype, &rhs_ctx.work_vec_b));
    PetscCall(VecDuplicate(prototype, &rhs_ctx.work_vec_c));
    return 0;
}

PetscErrorCode destroy_rhs_work_vectors(
    PetscCudaTsRhsContext& rhs_ctx)
{
    if (rhs_ctx.work_vec_c != nullptr) {
        PetscCall(VecDestroy(&rhs_ctx.work_vec_c));
    }
    if (rhs_ctx.work_vec_b != nullptr) {
        PetscCall(VecDestroy(&rhs_ctx.work_vec_b));
    }
    if (rhs_ctx.work_vec_a != nullptr) {
        PetscCall(VecDestroy(&rhs_ctx.work_vec_a));
    }
    return 0;
}

const CachedGroupedLayoutEntry* find_cached_grouped_layout(
    const std::vector<CachedGroupedLayoutEntry>& cached_grouped_layouts,
    const std::vector<Index>& sites)
{
    for (const CachedGroupedLayoutEntry& entry : cached_grouped_layouts) {
        if (same_sites(entry.sites, sites)) {
            return &entry;
        }
    }

    return nullptr;
}

struct BatchedDeviceStateBuffers {
    void* d_state = nullptr;
    void* d_accum = nullptr;
    void* d_term = nullptr;
    void* d_scaled_term = nullptr;
    std::size_t flat_bytes = 0;
};

struct BatchTsExecutionContext {
    TS ts = nullptr;
    Vec x = nullptr;
    Vec work_vec_a = nullptr;
    Vec work_vec_b = nullptr;
    const Solver* solver = nullptr;
    std::vector<CachedDissipatorAuxiliaries> cached_static_dissipators;
    std::vector<CachedGroupedLayoutEntry> cached_grouped_layouts;
    CuTensorExecutorCache executor_cache{};
    cudaStream_t stream = nullptr;
    Index batch_size = 0;
    Index total_size = 0;
};

void zero_device_buffer(
    void* d_ptr,
    std::size_t bytes,
    cudaStream_t stream);

void add_device_buffers(
    const void* d_a,
    const void* d_b,
    void* d_out,
    Index count,
    cudaStream_t stream);

void scale_device_buffer(
    const void* d_in,
    double scale,
    void* d_out,
    Index count,
    cudaStream_t stream);

void apply_grouped_commutator_cuda_batch(
    const Solver& solver,
    const std::string& term_name,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    const CudaGroupedStateLayout& cuda_grouped_layout,
    Index batch_size,
    CuTensorExecutorCache& executor_cache,
    const void* d_flat_input,
    void* d_flat_output,
    cudaStream_t consumer_stream);

void apply_grouped_dissipator_cuda_batch(
    const Solver& solver,
    const std::string& term_name,
    const std::vector<Complex>& l_op,
    const std::vector<Complex>& l_dag,
    const std::vector<Complex>& l_dag_l,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    const CudaGroupedStateLayout& cuda_grouped_layout,
    Index batch_size,
    CuTensorExecutorCache& executor_cache,
    const void* d_flat_input,
    void* d_flat_output,
    cudaStream_t consumer_stream);

void apply_grouped_commutator_cuda_single(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    const CudaGroupedStateLayout& cuda_grouped_layout,
    CuTensorExecutorCache& executor_cache,
    const void* d_flat_input,
    void* d_flat_output,
    cudaStream_t consumer_stream);

void apply_grouped_dissipator_cuda_single(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& local_op_dag,
    const std::vector<Complex>& local_op_dag_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    const CudaGroupedStateLayout& cuda_grouped_layout,
    CuTensorExecutorCache& executor_cache,
    const void* d_flat_input,
    void* d_flat_output,
    cudaStream_t consumer_stream);

void allocate_batched_device_state_buffers(
    Index density_dim,
    Index batch_size,
    BatchedDeviceStateBuffers& buffers)
{
    buffers.flat_bytes =
        static_cast<std::size_t>(density_dim) *
        static_cast<std::size_t>(batch_size) *
        sizeof(Complex);

    if (cudaMalloc(&buffers.d_state, buffers.flat_bytes) != cudaSuccess ||
        cudaMalloc(&buffers.d_accum, buffers.flat_bytes) != cudaSuccess ||
        cudaMalloc(&buffers.d_term, buffers.flat_bytes) != cudaSuccess ||
        cudaMalloc(&buffers.d_scaled_term, buffers.flat_bytes) != cudaSuccess) {
        throw std::runtime_error(
            "allocate_batched_device_state_buffers: device allocation failed");
    }
}

void free_batched_device_state_buffers(
    BatchedDeviceStateBuffers& buffers)
{
    if (buffers.d_scaled_term != nullptr) {
        cudaFree(buffers.d_scaled_term);
        buffers.d_scaled_term = nullptr;
    }
    if (buffers.d_term != nullptr) {
        cudaFree(buffers.d_term);
        buffers.d_term = nullptr;
    }
    if (buffers.d_accum != nullptr) {
        cudaFree(buffers.d_accum);
        buffers.d_accum = nullptr;
    }
    if (buffers.d_state != nullptr) {
        cudaFree(buffers.d_state);
        buffers.d_state = nullptr;
    }
    buffers.flat_bytes = 0;
}

std::vector<Complex> flatten_state_batch(
    const std::vector<std::vector<Complex>>& states,
    Index density_dim)
{
    std::vector<Complex> flat(
        static_cast<std::size_t>(states.size()) * static_cast<std::size_t>(density_dim),
        Complex{0.0, 0.0});

    for (Index batch = 0; batch < states.size(); ++batch) {
        if (states[batch].size() != density_dim) {
            throw std::invalid_argument(
                "flatten_state_batch: state size mismatch");
        }

        const std::size_t offset =
            static_cast<std::size_t>(batch) * static_cast<std::size_t>(density_dim);
        for (Index i = 0; i < density_dim; ++i) {
            flat[offset + static_cast<std::size_t>(i)] = states[batch][i];
        }
    }

    return flat;
}

void unflatten_state_batch(
    const std::vector<Complex>& flat_states,
    Index density_dim,
    std::vector<std::vector<Complex>>& states)
{
    const Index batch_size =
        static_cast<Index>(flat_states.size() / static_cast<std::size_t>(density_dim));

    states.resize(batch_size);
    for (Index batch = 0; batch < batch_size; ++batch) {
        states[batch].resize(density_dim);
        const std::size_t offset =
            static_cast<std::size_t>(batch) * static_cast<std::size_t>(density_dim);
        for (Index i = 0; i < density_dim; ++i) {
            states[batch][i] = flat_states[offset + static_cast<std::size_t>(i)];
        }
    }
}

PetscErrorCode create_batch_ts_execution_context(
    const Solver& solver,
    Index batch_size,
    BatchTsExecutionContext& ctx)
{
    ctx.solver = &solver;
    ctx.batch_size = batch_size;
    ctx.total_size = solver.layout.density_dim * batch_size;
    ctx.cached_static_dissipators = build_cached_static_dissipators(solver);
    ctx.cached_grouped_layouts = build_cached_grouped_layouts(solver);

    PetscCall(TSCreate(PETSC_COMM_SELF, &ctx.ts));
    PetscCall(TSSetType(ctx.ts, TSRK));

    PetscCall(VecCreate(PETSC_COMM_SELF, &ctx.x));
    PetscCall(VecSetSizes(ctx.x, PETSC_DECIDE, ctx.total_size));
    PetscCall(VecSetType(ctx.x, VECCUDA));
    PetscCall(VecSet(ctx.x, 0.0));

    PetscCall(VecDuplicate(ctx.x, &ctx.work_vec_a));
    PetscCall(VecDuplicate(ctx.x, &ctx.work_vec_b));

    if (cudaStreamCreate(&ctx.stream) != cudaSuccess) {
        return PETSC_ERR_LIB;
    }

    PetscCall(TSSetRHSFunction(
        ctx.ts,
        nullptr,
        [](TS, PetscReal t, Vec x, Vec f, void* raw_ctx) -> PetscErrorCode {
            auto* ctx = static_cast<BatchTsExecutionContext*>(raw_ctx);
            if (!ctx || !ctx->solver) {
                return PETSC_ERR_ARG_NULL;
            }

            const Solver& solver = *ctx->solver;
            const Index density_dim = solver.layout.density_dim;
            const Index total_size = ctx->total_size;
            const std::size_t total_bytes =
                static_cast<std::size_t>(total_size) * sizeof(Complex);

            const PetscScalar* d_x = nullptr;
            PetscScalar* d_f = nullptr;
            PetscScalar* d_term = nullptr;
            PetscScalar* d_scaled = nullptr;

#if defined(PETSC_HAVE_CUDA)
            PetscCall(VecCUDAGetArrayRead(x, &d_x));
            PetscCall(VecCUDAGetArrayWrite(f, &d_f));
            PetscCall(VecCUDAGetArrayWrite(ctx->work_vec_a, &d_term));
            PetscCall(VecCUDAGetArrayWrite(ctx->work_vec_b, &d_scaled));
#else
            PetscCall(VecGetArrayRead(x, &d_x));
            PetscCall(VecGetArray(f, &d_f));
            PetscCall(VecGetArray(ctx->work_vec_a, &d_term));
            PetscCall(VecGetArray(ctx->work_vec_b, &d_scaled));
#endif

            try {
                zero_device_buffer(reinterpret_cast<void*>(d_f), total_bytes, ctx->stream);

                for (Index batch_idx = 0; batch_idx < ctx->batch_size; ++batch_idx) {
                    const std::size_t offset =
                        static_cast<std::size_t>(batch_idx) *
                        static_cast<std::size_t>(density_dim);
                    const void* d_x_state =
                        reinterpret_cast<const Complex*>(d_x) + offset;
                    void* d_f_state =
                        reinterpret_cast<Complex*>(d_f) + offset;
                    void* d_term_state =
                        reinterpret_cast<Complex*>(d_term) + offset;
                    void* d_scaled_state =
                        reinterpret_cast<Complex*>(d_scaled) + offset;

                    for (const OperatorTerm& h_term : solver.model.hamiltonian_terms) {
                        const CachedGroupedLayoutEntry* layout_entry =
                            find_cached_grouped_layout(ctx->cached_grouped_layouts, h_term.sites);

                        if (layout_entry == nullptr) {
                            throw std::runtime_error(
                                "batch TS RHS: missing cached grouped Hamiltonian layout");
                        }

                        apply_grouped_commutator_cuda_single(
                            solver,
                            h_term.name,
                            h_term.matrix,
                            h_term.sites,
                            layout_entry->grouped_layout,
                            layout_entry->cuda_grouped_layout,
                            ctx->executor_cache,
                            d_x_state,
                            d_term_state,
                            ctx->stream);

                        add_device_buffers(
                            d_f_state,
                            d_term_state,
                            d_f_state,
                            density_dim,
                            ctx->stream);
                    }

                    for (const CachedDissipatorAuxiliaries& d_aux : ctx->cached_static_dissipators) {
                        const CachedGroupedLayoutEntry* layout_entry =
                            find_cached_grouped_layout(ctx->cached_grouped_layouts, d_aux.sites);

                        if (layout_entry == nullptr) {
                            throw std::runtime_error(
                                "batch TS RHS: missing cached grouped dissipator layout");
                        }

                        apply_grouped_dissipator_cuda_single(
                            solver,
                            d_aux.name,
                            d_aux.l_op,
                            d_aux.l_dag,
                            d_aux.l_dag_l,
                            d_aux.sites,
                            layout_entry->grouped_layout,
                            layout_entry->cuda_grouped_layout,
                            ctx->executor_cache,
                            d_x_state,
                            d_term_state,
                            ctx->stream);

                        add_device_buffers(
                            d_f_state,
                            d_term_state,
                            d_f_state,
                            density_dim,
                            ctx->stream);
                    }

                    for (const TimeDependentTerm& td_term : solver.model.time_dependent_hamiltonian_terms) {
                        const CachedGroupedLayoutEntry* layout_entry =
                            find_cached_grouped_layout(ctx->cached_grouped_layouts, td_term.sites);

                        if (layout_entry == nullptr) {
                            throw std::runtime_error(
                                "batch TS RHS: missing cached grouped time-dependent layout");
                        }

                        const double coeff =
                            evaluate_time_dependent_coefficient(td_term, static_cast<double>(t));

                        apply_grouped_commutator_cuda_single(
                            solver,
                            td_term.name,
                            td_term.matrix,
                            td_term.sites,
                            layout_entry->grouped_layout,
                            layout_entry->cuda_grouped_layout,
                            ctx->executor_cache,
                            d_x_state,
                            d_term_state,
                            ctx->stream);

                        scale_device_buffer(
                            d_term_state,
                            coeff,
                            d_scaled_state,
                            density_dim,
                            ctx->stream);

                        add_device_buffers(
                            d_f_state,
                            d_scaled_state,
                            d_f_state,
                            density_dim,
                            ctx->stream);
                    }
                }

                if (cudaStreamSynchronize(ctx->stream) != cudaSuccess) {
                    throw std::runtime_error("batch TS RHS: stream synchronize failed");
                }
            } catch (...) {
#if defined(PETSC_HAVE_CUDA)
                PetscCall(VecCUDARestoreArrayWrite(ctx->work_vec_b, &d_scaled));
                PetscCall(VecCUDARestoreArrayWrite(ctx->work_vec_a, &d_term));
                PetscCall(VecCUDARestoreArrayWrite(f, &d_f));
                PetscCall(VecCUDARestoreArrayRead(x, &d_x));
#else
                PetscCall(VecRestoreArray(ctx->work_vec_b, &d_scaled));
                PetscCall(VecRestoreArray(ctx->work_vec_a, &d_term));
                PetscCall(VecRestoreArray(f, &d_f));
                PetscCall(VecRestoreArrayRead(x, &d_x));
#endif
                return PETSC_ERR_LIB;
            }

#if defined(PETSC_HAVE_CUDA)
            PetscCall(VecCUDARestoreArrayWrite(ctx->work_vec_b, &d_scaled));
            PetscCall(VecCUDARestoreArrayWrite(ctx->work_vec_a, &d_term));
            PetscCall(VecCUDARestoreArrayWrite(f, &d_f));
            PetscCall(VecCUDARestoreArrayRead(x, &d_x));
#else
            PetscCall(VecRestoreArray(ctx->work_vec_b, &d_scaled));
            PetscCall(VecRestoreArray(ctx->work_vec_a, &d_term));
            PetscCall(VecRestoreArray(f, &d_f));
            PetscCall(VecRestoreArrayRead(x, &d_x));
#endif
            return 0;
        },
        &ctx));

    return 0;
}

PetscErrorCode destroy_batch_ts_execution_context(
    BatchTsExecutionContext& ctx)
{
    if (ctx.work_vec_b != nullptr) {
        PetscCall(VecDestroy(&ctx.work_vec_b));
    }
    if (ctx.work_vec_a != nullptr) {
        PetscCall(VecDestroy(&ctx.work_vec_a));
    }
    if (ctx.x != nullptr) {
        PetscCall(VecDestroy(&ctx.x));
    }
    if (ctx.ts != nullptr) {
        PetscCall(TSDestroy(&ctx.ts));
    }
    if (ctx.stream != nullptr) {
        cudaStreamDestroy(ctx.stream);
        ctx.stream = nullptr;
    }
    (void)destroy_cutensor_executor_cache(ctx.executor_cache);
    destroy_cached_grouped_layouts(ctx.cached_grouped_layouts);
    ctx.cached_static_dissipators.clear();
    ctx.solver = nullptr;
    ctx.batch_size = 0;
    ctx.total_size = 0;
    return 0;
}

void upload_flat_state_batch(
    const std::vector<std::vector<Complex>>& states,
    Index density_dim,
    void* d_state,
    cudaStream_t stream)
{
    const std::vector<Complex> flat_states =
        flatten_state_batch(states, density_dim);

    if (cudaMemcpyAsync(
            d_state,
            flat_states.data(),
            flat_states.size() * sizeof(Complex),
            cudaMemcpyHostToDevice,
            stream) != cudaSuccess) {
        throw std::runtime_error(
            "upload_flat_state_batch: device upload failed");
    }
}

std::vector<std::vector<Complex>> download_flat_state_batch(
    Index density_dim,
    Index batch_size,
    const void* d_state,
    cudaStream_t stream)
{
    std::vector<Complex> flat_states(
        static_cast<std::size_t>(density_dim) * static_cast<std::size_t>(batch_size),
        Complex{0.0, 0.0});

    if (cudaMemcpyAsync(
            flat_states.data(),
            d_state,
            flat_states.size() * sizeof(Complex),
            cudaMemcpyDeviceToHost,
            stream) != cudaSuccess) {
        throw std::runtime_error(
            "download_flat_state_batch: device download failed");
    }

    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error(
            "download_flat_state_batch: stream synchronize failed");
    }

    std::vector<std::vector<Complex>> states;
    unflatten_state_batch(flat_states, density_dim, states);
    return states;
}

void zero_device_buffer(
    void* d_buffer,
    std::size_t bytes,
    cudaStream_t stream)
{
    if (cudaMemsetAsync(d_buffer, 0, bytes, stream) != cudaSuccess) {
        throw std::runtime_error("zero_device_buffer: memset failed");
    }
}

void add_device_buffers(
    const void* d_a,
    const void* d_b,
    void* d_out,
    Index size,
    cudaStream_t stream)
{
    if (!launch_vector_add_kernel(d_a, d_b, d_out, size, stream)) {
        throw std::runtime_error("add_device_buffers: vector add failed");
    }
}

void scale_device_buffer(
    const void* d_in,
    double scale,
    void* d_out,
    Index size,
    cudaStream_t stream)
{
    if (!launch_vector_scale_kernel(d_in, scale, d_out, size, stream)) {
        throw std::runtime_error("scale_device_buffer: vector scale failed");
    }
}

CuTensorExecutor* get_or_prepare_executor(
    CuTensorExecutorCache& executor_cache,
    const std::string& cache_key,
    const CuTensorContractionDesc& contraction_desc,
    const std::string& operator_tag,
    const std::vector<Complex>& local_op,
    std::size_t grouped_bytes)
{
    CuTensorExecutor* executor = nullptr;
    const bool cache_ok =
        get_or_create_cutensor_executor(
            executor_cache,
            cache_key,
            contraction_desc,
            local_op.size() * sizeof(Complex),
            grouped_bytes,
            grouped_bytes,
            executor);

    if (!cache_ok || executor == nullptr) {
        throw std::runtime_error(
            "get_or_prepare_executor: executor creation failed");
    }

    if (!ensure_cutensor_executor_operator(*executor, operator_tag, local_op)) {
        throw std::runtime_error(
            "get_or_prepare_executor: operator upload failed");
    }

    return executor;
}

void apply_grouped_cuda_batch_impl(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    const CudaGroupedStateLayout& cuda_grouped_layout,
    Index batch_size,
    CuTensorExecutorCache& executor_cache,
    const CuTensorContractionDesc& contraction_desc,
    const std::string& cache_key,
    const std::string& operator_tag,
    const void* d_flat_input,
    void* d_flat_output,
    cudaStream_t consumer_stream)
{
    (void)solver;
    (void)target_sites;

    const std::size_t grouped_bytes =
        static_cast<std::size_t>(grouped_layout.grouped_size) *
        static_cast<std::size_t>(batch_size) *
        sizeof(Complex);

    CuTensorExecutor* executor =
        get_or_prepare_executor(
            executor_cache,
            cache_key,
            contraction_desc,
            operator_tag,
            local_op,
            grouped_bytes);

    if (!launch_flat_batch_to_grouped_kernel(
            cuda_grouped_layout,
            d_flat_input,
            executor->d_input,
            batch_size,
            executor->stream)) {
        throw std::runtime_error(
            "apply_grouped_cuda_batch_impl: flat->grouped regroup failed");
    }

    zero_device_buffer(executor->d_output, grouped_bytes, executor->stream);

    if (!execute_cutensor_executor_device(*executor)) {
        throw std::runtime_error(
            "apply_grouped_cuda_batch_impl: cuTENSOR execution failed");
    }

    if (!launch_grouped_batch_to_flat_kernel(
            cuda_grouped_layout,
            executor->d_output,
            d_flat_output,
            batch_size,
            executor->stream)) {
        throw std::runtime_error(
            "apply_grouped_cuda_batch_impl: grouped->flat regroup failed");
    }

    if (!record_cutensor_executor_completion(*executor) ||
        !wait_for_cutensor_executor_completion(*executor, consumer_stream)) {
        throw std::runtime_error(
            "apply_grouped_cuda_batch_impl: stream dependency handoff failed");
    }
}

void apply_grouped_commutator_cuda_batch(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    const CudaGroupedStateLayout& cuda_grouped_layout,
    Index batch_size,
    CuTensorExecutorCache& executor_cache,
    const void* d_flat_input,
    void* d_flat_output,
    cudaStream_t consumer_stream)
{
    const std::size_t grouped_elements =
        static_cast<std::size_t>(grouped_layout.grouped_size) *
        static_cast<std::size_t>(batch_size);
    const std::size_t grouped_bytes = grouped_elements * sizeof(Complex);

    const CuTensorContractionDesc left_desc =
        make_batched_cutensor_left_contraction_desc(
            target_sites,
            solver.model.local_dims,
            batch_size);
    const CuTensorContractionDesc right_desc =
        make_batched_cutensor_right_contraction_desc(
            target_sites,
            solver.model.local_dims,
            batch_size);

    CuTensorExecutor* left_executor =
        get_or_prepare_executor(
            executor_cache,
            "batched_comm_left_apply_" + term_label,
            left_desc,
            "batched_comm_left_operator_" + term_label,
            local_op,
            grouped_bytes);
    CuTensorExecutor* right_executor =
        get_or_prepare_executor(
            executor_cache,
            "batched_comm_right_apply_" + term_label,
            right_desc,
            "batched_comm_right_operator_" + term_label,
            local_op,
            grouped_bytes);
    CuTensorExecutor* combine_executor =
        get_or_prepare_executor(
            executor_cache,
            "batched_comm_combine_buffer_" + term_label,
            left_desc,
            "batched_comm_combine_operator_" + term_label,
            local_op,
            grouped_bytes);

    if (!launch_flat_batch_to_grouped_kernel(
            cuda_grouped_layout,
            d_flat_input,
            left_executor->d_input,
            batch_size,
            left_executor->stream) ||
        !launch_flat_batch_to_grouped_kernel(
            cuda_grouped_layout,
            d_flat_input,
            right_executor->d_input,
            batch_size,
            right_executor->stream)) {
        throw std::runtime_error(
            "apply_grouped_commutator_cuda_batch: flat->grouped regroup failed");
    }

    zero_device_buffer(left_executor->d_output, grouped_bytes, left_executor->stream);
    zero_device_buffer(right_executor->d_output, grouped_bytes, right_executor->stream);
    zero_device_buffer(combine_executor->d_output, grouped_bytes, combine_executor->stream);

    if (!execute_cutensor_executor_device(*left_executor) ||
        !execute_cutensor_executor_device(*right_executor)) {
        throw std::runtime_error(
            "apply_grouped_commutator_cuda_batch: cuTENSOR execution failed");
    }

    if (!wait_for_cutensor_executor_completion(*left_executor, combine_executor->stream) ||
        !wait_for_cutensor_executor_completion(*right_executor, combine_executor->stream)) {
        throw std::runtime_error(
            "apply_grouped_commutator_cuda_batch: stream dependency wait failed");
    }

    if (!launch_commutator_combine_kernel(
            left_executor->d_output,
            right_executor->d_output,
            combine_executor->d_output,
            grouped_elements,
            combine_executor->stream)) {
        throw std::runtime_error(
            "apply_grouped_commutator_cuda_batch: combine kernel failed");
    }

    if (!launch_grouped_batch_to_flat_kernel(
            cuda_grouped_layout,
            combine_executor->d_output,
            d_flat_output,
            batch_size,
            combine_executor->stream)) {
        throw std::runtime_error(
            "apply_grouped_commutator_cuda_batch: grouped->flat regroup failed");
    }

    if (!record_cutensor_executor_completion(*combine_executor) ||
        !wait_for_cutensor_executor_completion(*combine_executor, consumer_stream)) {
        throw std::runtime_error(
            "apply_grouped_commutator_cuda_batch: final stream handoff failed");
    }
}

void apply_grouped_commutator_cuda_single(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    const CudaGroupedStateLayout& cuda_grouped_layout,
    CuTensorExecutorCache& executor_cache,
    const void* d_flat_input,
    void* d_flat_output,
    cudaStream_t consumer_stream)
{
    if (should_use_tiny_dense_path(solver, target_sites)) {
        const std::vector<Complex> full_op =
            embed_k_site_operator(local_op, target_sites, solver.model.local_dims);
        TinyDenseOperatorKernelArg op_arg{};
        build_tiny_dense_operator_arg(full_op, solver.layout.hilbert_dim, op_arg);

        if (!launch_tiny_dense_commutator_kernel(
                op_arg,
                d_flat_input,
                d_flat_output,
                consumer_stream)) {
            throw std::runtime_error(
                "apply_grouped_commutator_cuda_single: tiny dense kernel failed");
        }
        return;
    }

    const std::size_t grouped_bytes =
        static_cast<std::size_t>(grouped_layout.grouped_size) * sizeof(Complex);

    const CuTensorContractionDesc left_desc =
        make_cutensor_left_contraction_desc(target_sites, solver.model.local_dims);
    const CuTensorContractionDesc right_desc =
        make_cutensor_right_contraction_desc(target_sites, solver.model.local_dims);

    CuTensorExecutor* left_executor =
        get_or_prepare_executor(
            executor_cache,
            "statewise_comm_left_apply_" + term_label,
            left_desc,
            "statewise_comm_left_operator_" + term_label,
            local_op,
            grouped_bytes);
    CuTensorExecutor* right_executor =
        get_or_prepare_executor(
            executor_cache,
            "statewise_comm_right_apply_" + term_label,
            right_desc,
            "statewise_comm_right_operator_" + term_label,
            local_op,
            grouped_bytes);
    CuTensorExecutor* combine_executor =
        get_or_prepare_executor(
            executor_cache,
            "statewise_comm_combine_buffer_" + term_label,
            left_desc,
            "statewise_comm_combine_operator_" + term_label,
            local_op,
            grouped_bytes);

    if (!launch_flat_to_grouped_kernel(
            cuda_grouped_layout,
            d_flat_input,
            left_executor->d_input,
            left_executor->stream) ||
        !launch_flat_to_grouped_kernel(
            cuda_grouped_layout,
            d_flat_input,
            right_executor->d_input,
            right_executor->stream)) {
        throw std::runtime_error(
            "apply_grouped_commutator_cuda_single: flat->grouped regroup failed");
    }

    zero_device_buffer(left_executor->d_output, grouped_bytes, left_executor->stream);
    zero_device_buffer(right_executor->d_output, grouped_bytes, right_executor->stream);
    zero_device_buffer(combine_executor->d_output, grouped_bytes, combine_executor->stream);

    if (!execute_cutensor_executor_device(*left_executor) ||
        !execute_cutensor_executor_device(*right_executor)) {
        throw std::runtime_error(
            "apply_grouped_commutator_cuda_single: cuTENSOR execution failed");
    }

    if (!wait_for_cutensor_executor_completion(*left_executor, combine_executor->stream) ||
        !wait_for_cutensor_executor_completion(*right_executor, combine_executor->stream)) {
        throw std::runtime_error(
            "apply_grouped_commutator_cuda_single: stream dependency wait failed");
    }

    if (!launch_commutator_combine_kernel(
            left_executor->d_output,
            right_executor->d_output,
            combine_executor->d_output,
            grouped_layout.grouped_size,
            combine_executor->stream)) {
        throw std::runtime_error(
            "apply_grouped_commutator_cuda_single: combine kernel failed");
    }

    if (!launch_grouped_to_flat_kernel(
            cuda_grouped_layout,
            combine_executor->d_output,
            d_flat_output,
            combine_executor->stream)) {
        throw std::runtime_error(
            "apply_grouped_commutator_cuda_single: grouped->flat regroup failed");
    }

    if (!record_cutensor_executor_completion(*combine_executor) ||
        !wait_for_cutensor_executor_completion(*combine_executor, consumer_stream)) {
        throw std::runtime_error(
            "apply_grouped_commutator_cuda_single: final stream handoff failed");
    }
}

void apply_grouped_dissipator_cuda_batch(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& local_op_dag,
    const std::vector<Complex>& local_op_dag_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    const CudaGroupedStateLayout& cuda_grouped_layout,
    Index batch_size,
    CuTensorExecutorCache& executor_cache,
    const void* d_flat_input,
    void* d_flat_output,
    cudaStream_t consumer_stream)
{
    const std::size_t grouped_elements =
        static_cast<std::size_t>(grouped_layout.grouped_size) *
        static_cast<std::size_t>(batch_size);
    const std::size_t grouped_bytes = grouped_elements * sizeof(Complex);

    const CuTensorContractionDesc left_desc =
        make_batched_cutensor_left_contraction_desc(
            target_sites,
            solver.model.local_dims,
            batch_size);
    const CuTensorContractionDesc right_desc =
        make_batched_cutensor_right_contraction_desc(
            target_sites,
            solver.model.local_dims,
            batch_size);

    CuTensorExecutor* jump_left_executor =
        get_or_prepare_executor(
            executor_cache,
            "batched_diss_jump_left_" + term_label,
            left_desc,
            "batched_diss_L_" + term_label,
            local_op,
            grouped_bytes);
    CuTensorExecutor* jump_right_executor =
        get_or_prepare_executor(
            executor_cache,
            "batched_diss_jump_right_" + term_label,
            right_desc,
            "batched_diss_Ldag_" + term_label,
            local_op_dag,
            grouped_bytes);
    CuTensorExecutor* norm_left_executor =
        get_or_prepare_executor(
            executor_cache,
            "batched_diss_norm_left_" + term_label,
            left_desc,
            "batched_diss_LdagL_" + term_label,
            local_op_dag_op,
            grouped_bytes);
    CuTensorExecutor* norm_right_executor =
        get_or_prepare_executor(
            executor_cache,
            "batched_diss_norm_right_" + term_label,
            right_desc,
            "batched_diss_LdagL_" + term_label,
            local_op_dag_op,
            grouped_bytes);

    if (!launch_flat_batch_to_grouped_kernel(
            cuda_grouped_layout,
            d_flat_input,
            jump_left_executor->d_input,
            batch_size,
            jump_left_executor->stream) ||
        !launch_flat_batch_to_grouped_kernel(
            cuda_grouped_layout,
            d_flat_input,
            norm_left_executor->d_input,
            batch_size,
            norm_left_executor->stream) ||
        !launch_flat_batch_to_grouped_kernel(
            cuda_grouped_layout,
            d_flat_input,
            norm_right_executor->d_input,
            batch_size,
            norm_right_executor->stream)) {
        throw std::runtime_error(
            "apply_grouped_dissipator_cuda_batch: flat->grouped regroup failed");
    }

    zero_device_buffer(jump_left_executor->d_output, grouped_bytes, jump_left_executor->stream);
    zero_device_buffer(jump_right_executor->d_output, grouped_bytes, jump_right_executor->stream);
    zero_device_buffer(norm_left_executor->d_output, grouped_bytes, norm_left_executor->stream);
    zero_device_buffer(norm_right_executor->d_output, grouped_bytes, norm_right_executor->stream);

    if (!execute_cutensor_executor_device(*jump_left_executor)) {
        throw std::runtime_error(
            "apply_grouped_dissipator_cuda_batch: jump-left execution failed");
    }

    if (!copy_cutensor_executor_output_to_input(*jump_left_executor, *jump_right_executor) ||
        !execute_cutensor_executor_device(*jump_right_executor)) {
        throw std::runtime_error(
            "apply_grouped_dissipator_cuda_batch: jump-right execution failed");
    }

    if (!execute_cutensor_executor_device(*norm_left_executor) ||
        !execute_cutensor_executor_device(*norm_right_executor)) {
        throw std::runtime_error(
            "apply_grouped_dissipator_cuda_batch: norm execution failed");
    }

    if (!wait_for_cutensor_executor_completion(*jump_right_executor, norm_right_executor->stream) ||
        !wait_for_cutensor_executor_completion(*norm_left_executor, norm_right_executor->stream) ||
        !wait_for_cutensor_executor_completion(*norm_right_executor, norm_right_executor->stream)) {
        throw std::runtime_error(
            "apply_grouped_dissipator_cuda_batch: stream dependency wait failed");
    }

    if (!launch_dissipator_combine_kernel(
            jump_right_executor->d_output,
            norm_left_executor->d_output,
            norm_right_executor->d_output,
            norm_right_executor->d_output,
            grouped_elements,
            norm_right_executor->stream)) {
        throw std::runtime_error(
            "apply_grouped_dissipator_cuda_batch: combine kernel failed");
    }

    if (!launch_grouped_batch_to_flat_kernel(
            cuda_grouped_layout,
            norm_right_executor->d_output,
            d_flat_output,
            batch_size,
            norm_right_executor->stream)) {
        throw std::runtime_error(
            "apply_grouped_dissipator_cuda_batch: grouped->flat regroup failed");
    }

    if (!record_cutensor_executor_completion(*norm_right_executor) ||
        !wait_for_cutensor_executor_completion(*norm_right_executor, consumer_stream)) {
        throw std::runtime_error(
            "apply_grouped_dissipator_cuda_batch: final stream handoff failed");
    }
}

void apply_grouped_dissipator_cuda_single(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& local_op_dag,
    const std::vector<Complex>& local_op_dag_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    const CudaGroupedStateLayout& cuda_grouped_layout,
    CuTensorExecutorCache& executor_cache,
    const void* d_flat_input,
    void* d_flat_output,
    cudaStream_t consumer_stream)
{
    if (should_use_tiny_dense_path(solver, target_sites)) {
        const std::vector<Complex> full_jump_op =
            embed_k_site_operator(local_op, target_sites, solver.model.local_dims);
        const std::vector<Complex> full_norm_op =
            embed_k_site_operator(local_op_dag_op, target_sites, solver.model.local_dims);
        TinyDenseOperatorKernelArg jump_arg{};
        TinyDenseOperatorKernelArg norm_arg{};
        build_tiny_dense_operator_arg(full_jump_op, solver.layout.hilbert_dim, jump_arg);
        build_tiny_dense_operator_arg(full_norm_op, solver.layout.hilbert_dim, norm_arg);

        if (!launch_tiny_dense_dissipator_kernel(
                jump_arg,
                norm_arg,
                d_flat_input,
                d_flat_output,
                consumer_stream)) {
            throw std::runtime_error(
                "apply_grouped_dissipator_cuda_single: tiny dense kernel failed");
        }
        return;
    }

    const std::size_t grouped_bytes =
        static_cast<std::size_t>(grouped_layout.grouped_size) * sizeof(Complex);

    const CuTensorContractionDesc left_desc =
        make_cutensor_left_contraction_desc(target_sites, solver.model.local_dims);
    const CuTensorContractionDesc right_desc =
        make_cutensor_right_contraction_desc(target_sites, solver.model.local_dims);

    CuTensorExecutor* jump_left_executor =
        get_or_prepare_executor(
            executor_cache,
            "statewise_diss_jump_left_" + term_label,
            left_desc,
            "statewise_diss_L_" + term_label,
            local_op,
            grouped_bytes);
    CuTensorExecutor* jump_right_executor =
        get_or_prepare_executor(
            executor_cache,
            "statewise_diss_jump_right_" + term_label,
            right_desc,
            "statewise_diss_Ldag_" + term_label,
            local_op_dag,
            grouped_bytes);
    CuTensorExecutor* norm_left_executor =
        get_or_prepare_executor(
            executor_cache,
            "statewise_diss_norm_left_" + term_label,
            left_desc,
            "statewise_diss_LdagL_" + term_label,
            local_op_dag_op,
            grouped_bytes);
    CuTensorExecutor* norm_right_executor =
        get_or_prepare_executor(
            executor_cache,
            "statewise_diss_norm_right_" + term_label,
            right_desc,
            "statewise_diss_LdagL_" + term_label,
            local_op_dag_op,
            grouped_bytes);

    if (!launch_flat_to_grouped_kernel(
            cuda_grouped_layout,
            d_flat_input,
            jump_left_executor->d_input,
            jump_left_executor->stream) ||
        !launch_flat_to_grouped_kernel(
            cuda_grouped_layout,
            d_flat_input,
            norm_left_executor->d_input,
            norm_left_executor->stream) ||
        !launch_flat_to_grouped_kernel(
            cuda_grouped_layout,
            d_flat_input,
            norm_right_executor->d_input,
            norm_right_executor->stream)) {
        throw std::runtime_error(
            "apply_grouped_dissipator_cuda_single: flat->grouped regroup failed");
    }

    zero_device_buffer(jump_left_executor->d_output, grouped_bytes, jump_left_executor->stream);
    zero_device_buffer(jump_right_executor->d_output, grouped_bytes, jump_right_executor->stream);
    zero_device_buffer(norm_left_executor->d_output, grouped_bytes, norm_left_executor->stream);
    zero_device_buffer(norm_right_executor->d_output, grouped_bytes, norm_right_executor->stream);

    if (!execute_cutensor_executor_device(*jump_left_executor)) {
        throw std::runtime_error(
            "apply_grouped_dissipator_cuda_single: jump-left execution failed");
    }

    if (!copy_cutensor_executor_output_to_input(*jump_left_executor, *jump_right_executor) ||
        !execute_cutensor_executor_device(*jump_right_executor)) {
        throw std::runtime_error(
            "apply_grouped_dissipator_cuda_single: jump-right execution failed");
    }

    if (!execute_cutensor_executor_device(*norm_left_executor) ||
        !execute_cutensor_executor_device(*norm_right_executor)) {
        throw std::runtime_error(
            "apply_grouped_dissipator_cuda_single: norm execution failed");
    }

    if (!wait_for_cutensor_executor_completion(*jump_right_executor, norm_right_executor->stream) ||
        !wait_for_cutensor_executor_completion(*norm_left_executor, norm_right_executor->stream) ||
        !wait_for_cutensor_executor_completion(*norm_right_executor, norm_right_executor->stream)) {
        throw std::runtime_error(
            "apply_grouped_dissipator_cuda_single: stream dependency wait failed");
    }

    if (!launch_dissipator_combine_kernel(
            jump_right_executor->d_output,
            norm_left_executor->d_output,
            norm_right_executor->d_output,
            norm_right_executor->d_output,
            grouped_layout.grouped_size,
            norm_right_executor->stream)) {
        throw std::runtime_error(
            "apply_grouped_dissipator_cuda_single: combine kernel failed");
    }

    if (!launch_grouped_to_flat_kernel(
            cuda_grouped_layout,
            norm_right_executor->d_output,
            d_flat_output,
            norm_right_executor->stream)) {
        throw std::runtime_error(
            "apply_grouped_dissipator_cuda_single: grouped->flat regroup failed");
    }

    if (!record_cutensor_executor_completion(*norm_right_executor) ||
        !wait_for_cutensor_executor_completion(*norm_right_executor, consumer_stream)) {
        throw std::runtime_error(
            "apply_grouped_dissipator_cuda_single: final stream handoff failed");
    }
}

} // namespace

PetscErrorCode create_cuda_batch_execution_context(
    const Solver& solver,
    CudaBatchExecutionContext& ctx)
{
    ctx.ts = nullptr;
    ctx.x = nullptr;
    ctx.initialized = false;

    const std::vector<Index> seed_target_sites =
        choose_seed_target_sites(solver);

    PetscCall(TSCreate(PETSC_COMM_SELF, &ctx.ts));
    PetscCall(TSSetType(ctx.ts, TSEULER));

    PetscCall(VecCreate(PETSC_COMM_SELF, &ctx.x));
    PetscCall(VecSetSizes(ctx.x, PETSC_DECIDE, solver.layout.density_dim));
    PetscCall(VecSetType(ctx.x, VECCUDA));
    PetscCall(VecSet(ctx.x, 0.0));

    GroupedStateLayout grouped_layout =
        make_grouped_state_layout(seed_target_sites, solver.model.local_dims);

    CudaGroupedStateLayout cuda_grouped_layout{};
    if (!create_cuda_grouped_state_layout(grouped_layout, cuda_grouped_layout)) {
        PetscCall(VecDestroy(&ctx.x));
        PetscCall(TSDestroy(&ctx.ts));
        return PETSC_ERR_LIB;
    }

    ctx.rhs_ctx = PetscCudaTsRhsContext{
        &solver,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        seed_target_sites,
        grouped_layout,
        cuda_grouped_layout,
        CuTensorExecutorCache{},
        build_cached_static_dissipators(solver),
        build_cached_grouped_layouts(solver),
        nullptr,
        nullptr,
        nullptr,
        nullptr
    };

    if (cudaStreamCreate(&ctx.rhs_ctx.elementwise_stream) != cudaSuccess) {
        destroy_cached_grouped_layouts(ctx.rhs_ctx.cached_grouped_layouts);
        (void)destroy_cuda_grouped_state_layout(ctx.rhs_ctx.cuda_grouped_layout);
        PetscCall(VecDestroy(&ctx.x));
        PetscCall(TSDestroy(&ctx.ts));
        return PETSC_ERR_LIB;
    }

    PetscCall(create_rhs_work_vectors(ctx.x, ctx.rhs_ctx));

    PetscCall(TSSetRHSFunction(
        ctx.ts,
        nullptr,
        ts_rhs_function_cuda_full_model_liouvillian,
        &ctx.rhs_ctx));

    ctx.initialized = true;
    return 0;
}

PetscErrorCode destroy_cuda_batch_execution_context(
    CudaBatchExecutionContext& ctx)
{
    if (!ctx.initialized) {
        return 0;
    }

    PetscCall(destroy_rhs_work_vectors(ctx.rhs_ctx));
    (void)destroy_cutensor_executor_cache(ctx.rhs_ctx.executor_cache);
    destroy_cached_grouped_layouts(ctx.rhs_ctx.cached_grouped_layouts);
    (void)destroy_cuda_grouped_state_layout(ctx.rhs_ctx.cuda_grouped_layout);

    if (ctx.rhs_ctx.elementwise_stream != nullptr) {
        cudaStreamDestroy(ctx.rhs_ctx.elementwise_stream);
        ctx.rhs_ctx.elementwise_stream = nullptr;
    }

    if (ctx.x != nullptr) {
        PetscCall(VecDestroy(&ctx.x));
    }

    if (ctx.ts != nullptr) {
        PetscCall(TSDestroy(&ctx.ts));
    }

    ctx.initialized = false;
    return 0;
}

BatchEvolutionResult evolve_density_batch_cpu_reference(
    const Solver& solver,
    const std::vector<std::vector<Complex>>& initial_states,
    const BatchEvolutionConfig& config)
{
    if (config.batch_size == 0) {
        throw std::invalid_argument(
            "evolve_density_batch_cpu_reference: batch_size must be > 0");
    }

    if (config.num_steps == 0) {
        throw std::invalid_argument(
            "evolve_density_batch_cpu_reference: num_steps must be > 0");
    }

    BatchEvolutionResult result;
    result.final_states.resize(initial_states.size());

    const Index density_dim = solver.layout.density_dim;

    for (Index batch_begin = 0; batch_begin < initial_states.size(); batch_begin += config.batch_size) {
        const Index batch_end =
            std::min<Index>(batch_begin + config.batch_size, initial_states.size());

        for (Index state_idx = batch_begin; state_idx < batch_end; ++state_idx) {
            if (initial_states[state_idx].size() != density_dim) {
                throw std::invalid_argument(
                    "evolve_density_batch_cpu_reference: initial state size mismatch");
            }

            std::vector<Complex> state = initial_states[state_idx];
            std::vector<Complex> rhs(density_dim, Complex{0.0, 0.0});

            for (Index step = 0; step < config.num_steps; ++step) {
                const double t =
                    config.t0 + static_cast<double>(step) * config.dt;

                ConstStateBuffer in_buf{state.data(), state.size()};
                StateBuffer rhs_buf{rhs.data(), rhs.size()};

                apply_liouvillian_at_time(solver, t, in_buf, rhs_buf);

                for (Index i = 0; i < density_dim; ++i) {
                    state[i] += config.dt * rhs[i];
                }
            }

            result.final_states[state_idx] = std::move(state);
        }
    }

    return result;
}

BatchEvolutionResult evolve_density_batch_cuda_ts(
    const Solver& solver,
    const std::vector<std::vector<Complex>>& initial_states,
    const BatchEvolutionConfig& config)
{
    if (config.batch_size == 0) {
        throw std::invalid_argument(
            "evolve_density_batch_cuda_ts: batch_size must be > 0");
    }

    if (config.num_steps == 0) {
        throw std::invalid_argument(
            "evolve_density_batch_cuda_ts: num_steps must be > 0");
    }

    BatchEvolutionResult result;
    result.final_states.resize(initial_states.size());

    const Index density_dim = solver.layout.density_dim;

    try {
        for (Index batch_begin = 0; batch_begin < initial_states.size(); batch_begin += config.batch_size) {
            const Index batch_end =
                std::min<Index>(batch_begin + config.batch_size, initial_states.size());
            const Index chunk_size = batch_end - batch_begin;

            std::vector<std::vector<Complex>> state_chunk(
                initial_states.begin() + batch_begin,
                initial_states.begin() + batch_end);

            BatchTsExecutionContext ctx{};
            try {
                PetscCallAbort(
                    PETSC_COMM_SELF,
                    create_batch_ts_execution_context(solver, chunk_size, ctx));

                PetscScalar* x_ptr = nullptr;
#if defined(PETSC_HAVE_CUDA)
                PetscCallAbort(PETSC_COMM_SELF, VecCUDAGetArrayWrite(ctx.x, &x_ptr));
#else
                PetscCallAbort(PETSC_COMM_SELF, VecGetArray(ctx.x, &x_ptr));
#endif
                if (cudaMemcpy(
                        x_ptr,
                        flatten_state_batch(state_chunk, density_dim).data(),
                        static_cast<std::size_t>(ctx.total_size) * sizeof(Complex),
                        cudaMemcpyHostToDevice) != cudaSuccess) {
#if defined(PETSC_HAVE_CUDA)
                    PetscCallAbort(PETSC_COMM_SELF, VecCUDARestoreArrayWrite(ctx.x, &x_ptr));
#else
                    PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(ctx.x, &x_ptr));
#endif
                    throw std::runtime_error(
                        "evolve_density_batch_cuda_ts: failed to upload initial batch state");
                }
#if defined(PETSC_HAVE_CUDA)
                PetscCallAbort(PETSC_COMM_SELF, VecCUDARestoreArrayWrite(ctx.x, &x_ptr));
#else
                PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(ctx.x, &x_ptr));
#endif

                const double total_time =
                    config.dt * static_cast<double>(config.num_steps);
                const Index effective_num_steps =
                    density_dim <= 16
                        ? std::max<Index>(config.num_steps, static_cast<Index>(1000))
                        : config.num_steps;
                const double dt_initial =
                    total_time / static_cast<double>(effective_num_steps);
                TSAdapt adapt = nullptr;
                PetscCallAbort(PETSC_COMM_SELF, TSGetAdapt(ctx.ts, &adapt));
                PetscCallAbort(PETSC_COMM_SELF, TSAdaptSetType(adapt, TSADAPTNONE));
                PetscCallAbort(PETSC_COMM_SELF, TSSetMaxSteps(ctx.ts, effective_num_steps));
                PetscCallAbort(PETSC_COMM_SELF, TSSetExactFinalTime(
                    ctx.ts,
                    TS_EXACTFINALTIME_MATCHSTEP));
                PetscCallAbort(PETSC_COMM_SELF, TSSetTime(ctx.ts, config.t0));
                PetscCallAbort(PETSC_COMM_SELF, TSSetStepNumber(ctx.ts, 0));
                PetscCallAbort(PETSC_COMM_SELF, TSSetTimeStep(ctx.ts, dt_initial));
                PetscCallAbort(PETSC_COMM_SELF, TSSetMaxTime(
                    ctx.ts,
                    config.t0 + total_time));
                PetscCallAbort(PETSC_COMM_SELF, TSSetFromOptions(ctx.ts));
                PetscCallAbort(PETSC_COMM_SELF, TSSolve(ctx.ts, ctx.x));

                const PetscScalar* x_read_ptr = nullptr;
#if defined(PETSC_HAVE_CUDA)
                PetscCallAbort(PETSC_COMM_SELF, VecCUDAGetArrayRead(ctx.x, &x_read_ptr));
#else
                PetscCallAbort(PETSC_COMM_SELF, VecGetArrayRead(ctx.x, &x_read_ptr));
#endif
                std::vector<Complex> flat_final(ctx.total_size, Complex{0.0, 0.0});
                if (cudaMemcpy(
                        flat_final.data(),
                        x_read_ptr,
                        static_cast<std::size_t>(ctx.total_size) * sizeof(Complex),
                        cudaMemcpyDeviceToHost) != cudaSuccess) {
#if defined(PETSC_HAVE_CUDA)
                    PetscCallAbort(PETSC_COMM_SELF, VecCUDARestoreArrayRead(ctx.x, &x_read_ptr));
#else
                    PetscCallAbort(PETSC_COMM_SELF, VecRestoreArrayRead(ctx.x, &x_read_ptr));
#endif
                    throw std::runtime_error(
                        "evolve_density_batch_cuda_ts: failed to download final batch state");
                }
                if (cudaDeviceSynchronize() != cudaSuccess) {
#if defined(PETSC_HAVE_CUDA)
                    PetscCallAbort(PETSC_COMM_SELF, VecCUDARestoreArrayRead(ctx.x, &x_read_ptr));
#else
                    PetscCallAbort(PETSC_COMM_SELF, VecRestoreArrayRead(ctx.x, &x_read_ptr));
#endif
                    throw std::runtime_error(
                        "evolve_density_batch_cuda_ts: device synchronize failed after solve");
                }
#if defined(PETSC_HAVE_CUDA)
                PetscCallAbort(PETSC_COMM_SELF, VecCUDARestoreArrayRead(ctx.x, &x_read_ptr));
#else
                PetscCallAbort(PETSC_COMM_SELF, VecRestoreArrayRead(ctx.x, &x_read_ptr));
#endif

                std::vector<std::vector<Complex>> final_states;
                unflatten_state_batch(flat_final, density_dim, final_states);
                for (Index state_idx = 0; state_idx < chunk_size; ++state_idx) {
                    result.final_states[batch_begin + state_idx] = final_states[state_idx];
                }
            } catch (...) {
                PetscCallAbort(PETSC_COMM_SELF, destroy_batch_ts_execution_context(ctx));
                throw;
            }

            PetscCallAbort(PETSC_COMM_SELF, destroy_batch_ts_execution_context(ctx));
        }
    } catch (...) {
        throw;
    }
    return result;
}

BatchEvolutionTiming time_density_batch_cpu_reference(
    const Solver& solver,
    const std::vector<std::vector<Complex>>& initial_states,
    const BatchEvolutionConfig& config)
{
    const auto t_start = std::chrono::steady_clock::now();

    BatchEvolutionResult result =
        evolve_density_batch_cpu_reference(
            solver,
            initial_states,
            config);

    const auto t_end = std::chrono::steady_clock::now();

    BatchEvolutionTiming timing;
    timing.elapsed_seconds =
        std::chrono::duration<double>(t_end - t_start).count();
    timing.gpu_elapsed_seconds = 0.0;
    timing.result = std::move(result);
    return timing;
}

BatchEvolutionTiming time_density_batch_cuda_ts(
    const Solver& solver,
    const std::vector<std::vector<Complex>>& initial_states,
    const BatchEvolutionConfig& config)
{
    const auto t_start = std::chrono::steady_clock::now();

    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;

    if (cudaEventCreate(&gpu_start) != cudaSuccess) {
        throw std::runtime_error("time_density_batch_cuda_ts: cudaEventCreate start failed");
    }

    if (cudaEventCreate(&gpu_stop) != cudaSuccess) {
        cudaEventDestroy(gpu_start);
        throw std::runtime_error("time_density_batch_cuda_ts: cudaEventCreate stop failed");
    }

    if (cudaEventRecord(gpu_start, 0) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error("time_density_batch_cuda_ts: cudaEventRecord start failed");
    }

    BatchEvolutionResult result =
        evolve_density_batch_cuda_ts(
            solver,
            initial_states,
            config);

    if (cudaEventRecord(gpu_stop, 0) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error("time_density_batch_cuda_ts: cudaEventRecord stop failed");
    }

    if (cudaEventSynchronize(gpu_stop) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error("time_density_batch_cuda_ts: cudaEventSynchronize failed");
    }

    float gpu_milliseconds = 0.0f;
    if (cudaEventElapsedTime(&gpu_milliseconds, gpu_start, gpu_stop) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error("time_density_batch_cuda_ts: cudaEventElapsedTime failed");
    }

    cudaEventDestroy(gpu_stop);
    cudaEventDestroy(gpu_start);

    const auto t_end = std::chrono::steady_clock::now();

    BatchEvolutionTiming timing;
    timing.elapsed_seconds =
        std::chrono::duration<double>(t_end - t_start).count();
    timing.gpu_elapsed_seconds =
        static_cast<double>(gpu_milliseconds) * 1.0e-3;
    timing.result = std::move(result);
    return timing;
}

} // namespace culindblad

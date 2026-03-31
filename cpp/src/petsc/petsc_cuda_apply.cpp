#include "culindblad/petsc_cuda_apply.hpp"

#include <petscvec.h>

#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "culindblad/cuda_elementwise.hpp"
#include "culindblad/cuda_grouped_layout.hpp"
#include "culindblad/cutensor_contraction_desc.hpp"
#include "culindblad/cutensor_executor.hpp"
#include "culindblad/cutensor_executor_cache.hpp"
#include "culindblad/grouped_state_layout.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

namespace {

PetscErrorCode get_petsc_vec_device_read_ptr(
    Vec x,
    const void*& d_ptr_out)
{
    const PetscScalar* x_ptr = nullptr;

#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDAGetArrayRead(x, &x_ptr));
#else
    PetscCall(VecGetArrayRead(x, &x_ptr));
#endif

    d_ptr_out = reinterpret_cast<const void*>(x_ptr);
    return 0;
}

PetscErrorCode restore_petsc_vec_device_read_ptr(
    Vec x,
    const void*& d_ptr_in)
{
    const PetscScalar* x_ptr =
        reinterpret_cast<const PetscScalar*>(d_ptr_in);

#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDARestoreArrayRead(x, &x_ptr));
#else
    PetscCall(VecRestoreArrayRead(x, &x_ptr));
#endif

    d_ptr_in = nullptr;
    return 0;
}

PetscErrorCode get_petsc_vec_device_write_ptr(
    Vec y,
    void*& d_ptr_out)
{
    PetscScalar* y_ptr = nullptr;

#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDAGetArray(y, &y_ptr));
#else
    PetscCall(VecGetArray(y, &y_ptr));
#endif

    d_ptr_out = reinterpret_cast<void*>(y_ptr);
    return 0;
}

PetscErrorCode restore_petsc_vec_device_write_ptr(
    Vec y,
    void*& d_ptr_in)
{
    PetscScalar* y_ptr =
        reinterpret_cast<PetscScalar*>(d_ptr_in);

#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDARestoreArray(y, &y_ptr));
#else
    PetscCall(VecRestoreArray(y, &y_ptr));
#endif

    d_ptr_in = nullptr;
    return 0;
}

PetscErrorCode get_or_prepare_executor(
    CuTensorExecutorCache& executor_cache,
    const std::string& cache_key,
    const CuTensorContractionDesc& contraction_desc,
    const std::string& operator_tag,
    const std::vector<Complex>& local_op,
    std::size_t grouped_bytes,
    CuTensorExecutor*& executor)
{
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
        return PETSC_ERR_LIB;
    }

    const bool op_ok =
        ensure_cutensor_executor_operator(
            *executor,
            operator_tag,
            local_op);

    if (!op_ok) {
        return PETSC_ERR_LIB;
    }

    return 0;
}

PetscErrorCode apply_grouped_cuda_vec_impl(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    const CudaGroupedStateLayout& cuda_grouped_layout,
    CuTensorExecutorCache& executor_cache,
    const CuTensorContractionDesc& contraction_desc,
    const std::string& cache_key,
    const std::string& operator_tag,
    Vec x,
    Vec y,
    cudaStream_t consumer_stream)
{
    (void)solver;
    (void)target_sites;

    const std::size_t grouped_bytes =
        grouped_layout.grouped_size * sizeof(Complex);

    const void* d_flat_input = nullptr;
    void* d_flat_output = nullptr;

    PetscCall(get_petsc_vec_device_read_ptr(x, d_flat_input));
    PetscCall(get_petsc_vec_device_write_ptr(y, d_flat_output));

    CuTensorExecutor* executor = nullptr;
    PetscErrorCode ierr = get_or_prepare_executor(
        executor_cache,
        cache_key,
        contraction_desc,
        operator_tag,
        local_op,
        grouped_bytes,
        executor);
    if (ierr != 0) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return ierr;
    }

    const bool regroup_in_ok =
        launch_flat_to_grouped_kernel(
            cuda_grouped_layout,
            d_flat_input,
            executor->d_input,
            executor->stream);

    if (!regroup_in_ok) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    if (cudaMemsetAsync(
            executor->d_output,
            0,
            grouped_bytes,
            executor->stream) != cudaSuccess) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    const bool exec_ok =
        execute_cutensor_executor_device(*executor);

    if (!exec_ok) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    const bool regroup_out_ok =
        launch_grouped_to_flat_kernel(
            cuda_grouped_layout,
            executor->d_output,
            d_flat_output,
            executor->stream);

    if (!regroup_out_ok) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    if (!record_cutensor_executor_completion(*executor)) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    if (consumer_stream != nullptr) {
        if (!wait_for_cutensor_executor_completion(*executor, consumer_stream)) {
            PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
            PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
            return PETSC_ERR_LIB;
        }
    } else if (cudaStreamSynchronize(executor->stream) != cudaSuccess) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
    PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
    return 0;
}

} // namespace

PetscErrorCode apply_grouped_left_cuda_vec(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    const CudaGroupedStateLayout& cuda_grouped_layout,
    CuTensorExecutorCache& executor_cache,
    Vec x,
    Vec y,
    cudaStream_t consumer_stream)
{
    const CuTensorContractionDesc left_desc =
        make_cutensor_left_contraction_desc(target_sites, solver.model.local_dims);

    return apply_grouped_cuda_vec_impl(
        solver,
        local_op,
        target_sites,
        grouped_layout,
        cuda_grouped_layout,
        executor_cache,
        left_desc,
        "petsc_grouped_left_apply_" + term_label,
        "petsc_grouped_left_operator_" + term_label,
        x,
        y,
        consumer_stream);
}

PetscErrorCode apply_grouped_right_cuda_vec(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    const CudaGroupedStateLayout& cuda_grouped_layout,
    CuTensorExecutorCache& executor_cache,
    Vec x,
    Vec y,
    cudaStream_t consumer_stream)
{
    const CuTensorContractionDesc right_desc =
        make_cutensor_right_contraction_desc(target_sites, solver.model.local_dims);

    return apply_grouped_cuda_vec_impl(
        solver,
        local_op,
        target_sites,
        grouped_layout,
        cuda_grouped_layout,
        executor_cache,
        right_desc,
        "petsc_grouped_right_apply_" + term_label,
        "petsc_grouped_right_operator_" + term_label,
        x,
        y,
        consumer_stream);
}

PetscErrorCode apply_grouped_commutator_cuda_vec(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    const CudaGroupedStateLayout& cuda_grouped_layout,
    CuTensorExecutorCache& executor_cache,
    Vec x,
    Vec y,
    cudaStream_t consumer_stream)
{
    const std::size_t grouped_bytes =
        grouped_layout.grouped_size * sizeof(Complex);

    const void* d_flat_input = nullptr;
    void* d_flat_output = nullptr;

    PetscCall(get_petsc_vec_device_read_ptr(x, d_flat_input));
    PetscCall(get_petsc_vec_device_write_ptr(y, d_flat_output));

    const CuTensorContractionDesc left_desc =
        make_cutensor_left_contraction_desc(target_sites, solver.model.local_dims);
    const CuTensorContractionDesc right_desc =
        make_cutensor_right_contraction_desc(target_sites, solver.model.local_dims);

    CuTensorExecutor* left_executor = nullptr;
    CuTensorExecutor* right_executor = nullptr;
    CuTensorExecutor* combine_executor = nullptr;

    PetscErrorCode ierr = get_or_prepare_executor(
        executor_cache,
        "petsc_grouped_comm_left_apply_" + term_label,
        left_desc,
        "petsc_grouped_comm_left_operator_" + term_label,
        local_op,
        grouped_bytes,
        left_executor);
    if (ierr != 0) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return ierr;
    }

    ierr = get_or_prepare_executor(
        executor_cache,
        "petsc_grouped_comm_right_apply_" + term_label,
        right_desc,
        "petsc_grouped_comm_right_operator_" + term_label,
        local_op,
        grouped_bytes,
        right_executor);
    if (ierr != 0) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return ierr;
    }

    ierr = get_or_prepare_executor(
        executor_cache,
        "petsc_grouped_comm_combine_buffer_" + term_label,
        left_desc,
        "petsc_grouped_comm_combine_operator_" + term_label,
        local_op,
        grouped_bytes,
        combine_executor);
    if (ierr != 0) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return ierr;
    }

    const bool regroup_left_ok =
        launch_flat_to_grouped_kernel(
            cuda_grouped_layout,
            d_flat_input,
            left_executor->d_input,
            left_executor->stream);

    const bool regroup_right_ok =
        launch_flat_to_grouped_kernel(
            cuda_grouped_layout,
            d_flat_input,
            right_executor->d_input,
            right_executor->stream);

    if (!regroup_left_ok || !regroup_right_ok) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    if (cudaMemsetAsync(left_executor->d_output, 0, grouped_bytes, left_executor->stream) != cudaSuccess ||
        cudaMemsetAsync(right_executor->d_output, 0, grouped_bytes, right_executor->stream) != cudaSuccess ||
        cudaMemsetAsync(combine_executor->d_output, 0, grouped_bytes, combine_executor->stream) != cudaSuccess) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    const bool left_ok =
        execute_cutensor_executor_device(*left_executor);
    const bool right_ok =
        execute_cutensor_executor_device(*right_executor);

    if (!left_ok || !right_ok) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    if (!wait_for_cutensor_executor_completion(*left_executor, combine_executor->stream) ||
        !wait_for_cutensor_executor_completion(*right_executor, combine_executor->stream)) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    const bool combine_ok =
        launch_commutator_combine_kernel(
            left_executor->d_output,
            right_executor->d_output,
            combine_executor->d_output,
            grouped_layout.grouped_size,
            combine_executor->stream);

    if (!combine_ok) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    const bool regroup_out_ok =
        launch_grouped_to_flat_kernel(
            cuda_grouped_layout,
            combine_executor->d_output,
            d_flat_output,
            combine_executor->stream);

    if (!regroup_out_ok) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    if (!record_cutensor_executor_completion(*combine_executor)) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    if (consumer_stream != nullptr) {
        if (!wait_for_cutensor_executor_completion(*combine_executor, consumer_stream)) {
            PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
            PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
            return PETSC_ERR_LIB;
        }
    } else if (cudaStreamSynchronize(combine_executor->stream) != cudaSuccess) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
    PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
    return 0;
}

PetscErrorCode apply_grouped_dissipator_cuda_vec(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& local_op_dag,
    const std::vector<Complex>& local_op_dag_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    const CudaGroupedStateLayout& cuda_grouped_layout,
    CuTensorExecutorCache& executor_cache,
    Vec x,
    Vec y,
    cudaStream_t consumer_stream)
{
    const std::size_t grouped_bytes =
        grouped_layout.grouped_size * sizeof(Complex);

    const void* d_flat_input = nullptr;
    void* d_flat_output = nullptr;

    PetscCall(get_petsc_vec_device_read_ptr(x, d_flat_input));
    PetscCall(get_petsc_vec_device_write_ptr(y, d_flat_output));

    const CuTensorContractionDesc left_desc =
        make_cutensor_left_contraction_desc(target_sites, solver.model.local_dims);
    const CuTensorContractionDesc right_desc =
        make_cutensor_right_contraction_desc(target_sites, solver.model.local_dims);

    CuTensorExecutor* jump_left_executor = nullptr;
    CuTensorExecutor* jump_right_executor = nullptr;
    CuTensorExecutor* norm_left_executor = nullptr;
    CuTensorExecutor* norm_right_executor = nullptr;

    PetscErrorCode ierr = get_or_prepare_executor(
        executor_cache,
        "petsc_grouped_diss_jump_left_" + term_label,
        left_desc,
        "petsc_grouped_diss_L_" + term_label,
        local_op,
        grouped_bytes,
        jump_left_executor);
    if (ierr != 0) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return ierr;
    }

    ierr = get_or_prepare_executor(
        executor_cache,
        "petsc_grouped_diss_jump_right_" + term_label,
        right_desc,
        "petsc_grouped_diss_Ldag_" + term_label,
        local_op_dag,
        grouped_bytes,
        jump_right_executor);
    if (ierr != 0) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return ierr;
    }

    ierr = get_or_prepare_executor(
        executor_cache,
        "petsc_grouped_diss_norm_left_" + term_label,
        left_desc,
        "petsc_grouped_diss_LdagL_" + term_label,
        local_op_dag_op,
        grouped_bytes,
        norm_left_executor);
    if (ierr != 0) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return ierr;
    }

    ierr = get_or_prepare_executor(
        executor_cache,
        "petsc_grouped_diss_norm_right_" + term_label,
        right_desc,
        "petsc_grouped_diss_LdagL_" + term_label,
        local_op_dag_op,
        grouped_bytes,
        norm_right_executor);
    if (ierr != 0) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return ierr;
    }

    const bool regroup_jump_left_ok =
        launch_flat_to_grouped_kernel(
            cuda_grouped_layout,
            d_flat_input,
            jump_left_executor->d_input,
            jump_left_executor->stream);

    const bool regroup_norm_left_ok =
        launch_flat_to_grouped_kernel(
            cuda_grouped_layout,
            d_flat_input,
            norm_left_executor->d_input,
            norm_left_executor->stream);

    const bool regroup_norm_right_ok =
        launch_flat_to_grouped_kernel(
            cuda_grouped_layout,
            d_flat_input,
            norm_right_executor->d_input,
            norm_right_executor->stream);

    if (!regroup_jump_left_ok || !regroup_norm_left_ok || !regroup_norm_right_ok) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    if (cudaMemsetAsync(jump_left_executor->d_output, 0, grouped_bytes, jump_left_executor->stream) != cudaSuccess ||
        cudaMemsetAsync(jump_right_executor->d_output, 0, grouped_bytes, jump_right_executor->stream) != cudaSuccess ||
        cudaMemsetAsync(norm_left_executor->d_output, 0, grouped_bytes, norm_left_executor->stream) != cudaSuccess ||
        cudaMemsetAsync(norm_right_executor->d_output, 0, grouped_bytes, norm_right_executor->stream) != cudaSuccess) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    const bool jump_left_ok =
        execute_cutensor_executor_device(*jump_left_executor);

    if (!jump_left_ok) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    const bool jump_right_ok =
        copy_cutensor_executor_output_to_input(*jump_left_executor, *jump_right_executor) &&
        execute_cutensor_executor_device(*jump_right_executor);

    if (!jump_right_ok) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    const bool norm_left_ok =
        execute_cutensor_executor_device(*norm_left_executor);
    const bool norm_right_ok =
        execute_cutensor_executor_device(*norm_right_executor);

    if (!norm_left_ok || !norm_right_ok) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    if (!wait_for_cutensor_executor_completion(*jump_right_executor, norm_right_executor->stream) ||
        !wait_for_cutensor_executor_completion(*norm_left_executor, norm_right_executor->stream) ||
        !wait_for_cutensor_executor_completion(*norm_right_executor, norm_right_executor->stream)) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    const bool combine_ok =
        launch_dissipator_combine_kernel(
            jump_right_executor->d_output,
            norm_left_executor->d_output,
            norm_right_executor->d_output,
            norm_right_executor->d_output,
            grouped_layout.grouped_size,
            norm_right_executor->stream);

    if (!combine_ok) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    const bool regroup_out_ok =
        launch_grouped_to_flat_kernel(
            cuda_grouped_layout,
            norm_right_executor->d_output,
            d_flat_output,
            norm_right_executor->stream);

    if (!regroup_out_ok) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    if (!record_cutensor_executor_completion(*norm_right_executor)) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    if (consumer_stream != nullptr) {
        if (!wait_for_cutensor_executor_completion(*norm_right_executor, consumer_stream)) {
            PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
            PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
            return PETSC_ERR_LIB;
        }
    } else if (cudaStreamSynchronize(norm_right_executor->stream) != cudaSuccess) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
    PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
    return 0;
}

} // namespace culindblad

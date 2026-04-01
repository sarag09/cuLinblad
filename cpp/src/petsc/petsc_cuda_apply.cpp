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

bool wait_for_executor_event(
    CuTensorExecutor& producer,
    cudaStream_t consumer_stream)
{
    if (consumer_stream == nullptr) {
        return producer.stream != nullptr &&
               cudaStreamSynchronize(producer.stream) == cudaSuccess;
    }

    return wait_for_cutensor_executor_completion(producer, consumer_stream);
}

bool wait_for_stream_event(
    cudaStream_t producer_stream,
    cudaStream_t consumer_stream)
{
    if (producer_stream == nullptr || consumer_stream == nullptr) {
        return false;
    }

    if (producer_stream == consumer_stream) {
        return true;
    }

    cudaEvent_t completion_event = nullptr;
    if (cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming) != cudaSuccess) {
        return false;
    }

    const cudaError_t record_status =
        cudaEventRecord(completion_event, producer_stream);
    const cudaError_t wait_status =
        record_status == cudaSuccess
            ? cudaStreamWaitEvent(consumer_stream, completion_event, 0)
            : cudaErrorUnknown;
    const cudaError_t destroy_status = cudaEventDestroy(completion_event);

    return record_status == cudaSuccess &&
           wait_status == cudaSuccess &&
           destroy_status == cudaSuccess;
}

bool wait_for_cuda_event(
    cudaEvent_t completion_event,
    cudaStream_t consumer_stream)
{
    if (completion_event == nullptr) {
        return false;
    }

    if (consumer_stream == nullptr) {
        return cudaEventSynchronize(completion_event) == cudaSuccess;
    }

    return cudaStreamWaitEvent(consumer_stream, completion_event, 0) == cudaSuccess;
}

bool copy_executor_input_from_executor(
    CuTensorExecutor& src_executor,
    CuTensorExecutor& dst_executor)
{
    if (!wait_for_cutensor_executor_completion(src_executor, dst_executor.stream)) {
        return false;
    }

    return cudaMemcpyAsync(
               dst_executor.d_input,
               src_executor.d_input,
               dst_executor.input_bytes,
               cudaMemcpyDeviceToDevice,
               dst_executor.stream) == cudaSuccess;
}

bool copy_device_buffer_to_executor_input(
    const void* d_input,
    std::size_t input_bytes,
    cudaEvent_t input_ready_event,
    cudaStream_t producer_stream,
    CuTensorExecutor& dst_executor)
{
    const bool input_wait_ok =
        input_ready_event != nullptr
            ? wait_for_cuda_event(input_ready_event, dst_executor.stream)
            : (producer_stream == nullptr ||
               wait_for_stream_event(producer_stream, dst_executor.stream));
    if (!input_wait_ok) {
        return false;
    }

    return cudaMemcpyAsync(
               dst_executor.d_input,
               d_input,
               input_bytes,
               cudaMemcpyDeviceToDevice,
               dst_executor.stream) == cudaSuccess;
}

struct ExecutorInputBinding {
    CuTensorExecutor* executor = nullptr;
    void* original_d_input = nullptr;
};

bool bind_device_buffer_as_executor_input(
    const void* d_input,
    std::size_t input_bytes,
    cudaEvent_t input_ready_event,
    cudaStream_t producer_stream,
    CuTensorExecutor& dst_executor,
    ExecutorInputBinding& binding)
{
    if (d_input == nullptr || input_bytes != dst_executor.input_bytes) {
        return false;
    }

    const bool input_wait_ok =
        input_ready_event != nullptr
            ? wait_for_cuda_event(input_ready_event, dst_executor.stream)
            : (producer_stream == nullptr ||
               wait_for_stream_event(producer_stream, dst_executor.stream));
    if (!input_wait_ok) {
        return false;
    }

    binding.executor = &dst_executor;
    binding.original_d_input = dst_executor.d_input;
    dst_executor.d_input = const_cast<void*>(d_input);
    return true;
}

bool bind_executor_output_as_executor_input(
    CuTensorExecutor& src_executor,
    CuTensorExecutor& dst_executor,
    ExecutorInputBinding& binding)
{
    if (src_executor.d_output == nullptr ||
        src_executor.output_bytes != dst_executor.input_bytes) {
        return false;
    }

    if (!wait_for_cutensor_executor_completion(src_executor, dst_executor.stream)) {
        return false;
    }

    binding.executor = &dst_executor;
    binding.original_d_input = dst_executor.d_input;
    dst_executor.d_input = src_executor.d_output;
    return true;
}

void restore_executor_input_binding(
    ExecutorInputBinding& binding)
{
    if (binding.executor != nullptr) {
        binding.executor->d_input = binding.original_d_input;
    }

    binding.executor = nullptr;
    binding.original_d_input = nullptr;
}

bool zero_executor_output(
    CuTensorExecutor& executor,
    std::size_t output_bytes)
{
    return cudaMemsetAsync(
               executor.d_output,
               0,
               output_bytes,
               executor.stream) == cudaSuccess;
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
    Index batch_size,
    cudaStream_t consumer_stream)
{
    (void)solver;
    (void)target_sites;

    const std::size_t grouped_bytes =
        batch_size * grouped_layout.grouped_size * sizeof(Complex);

    const void* d_flat_input = nullptr;
    void* d_flat_output = nullptr;

    PetscCall(get_petsc_vec_device_read_ptr(x, d_flat_input));
    PetscCall(get_petsc_vec_device_write_ptr(y, d_flat_output));

    CuTensorExecutor* executor = nullptr;
    PetscErrorCode ierr = get_or_prepare_executor(
        executor_cache,
        cache_key + "_batch_" + std::to_string(batch_size),
        contraction_desc,
        operator_tag + "_batch_" + std::to_string(batch_size),
        local_op,
        grouped_bytes,
        executor);
    if (ierr != 0) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return ierr;
    }

    const bool regroup_in_ok =
        launch_flat_to_grouped_batched_kernel(
            cuda_grouped_layout,
            batch_size,
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
        launch_grouped_to_flat_batched_kernel(
            cuda_grouped_layout,
            batch_size,
            executor->d_output,
            d_flat_output,
            executor->stream);

    if (!regroup_out_ok) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    if (!record_cutensor_executor_completion(*executor) ||
        !wait_for_executor_event(*executor, consumer_stream)) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
    PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
    return 0;
}

PetscErrorCode flatten_grouped_buffer_to_device_ptr(
    const CudaGroupedStateLayout& cuda_grouped_layout,
    const void* d_grouped_input,
    void* d_flat_output,
    Index batch_size,
    cudaStream_t consumer_stream,
    cudaStream_t producer_stream)
{
    const bool regroup_out_ok =
        launch_grouped_to_flat_batched_kernel(
            cuda_grouped_layout,
            batch_size,
            d_grouped_input,
            d_flat_output,
            producer_stream);

    if (!regroup_out_ok) {
        return PETSC_ERR_LIB;
    }

    if (producer_stream == consumer_stream) {
        return 0;
    }

    if (consumer_stream == nullptr) {
        if (cudaStreamSynchronize(producer_stream) != cudaSuccess) {
            return PETSC_ERR_LIB;
        }
        return 0;
    }

    cudaEvent_t completion_event = nullptr;
    if (cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming) != cudaSuccess) {
        return PETSC_ERR_LIB;
    }

    if (cudaEventRecord(completion_event, producer_stream) != cudaSuccess) {
        cudaEventDestroy(completion_event);
        return PETSC_ERR_LIB;
    }

    const cudaError_t wait_status =
        cudaStreamWaitEvent(consumer_stream, completion_event, 0);
    const cudaError_t destroy_status = cudaEventDestroy(completion_event);

    if (wait_status != cudaSuccess || destroy_status != cudaSuccess) {
        return PETSC_ERR_LIB;
    }

    return 0;
}

PetscErrorCode apply_grouped_commutator_cuda_buffer_impl(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    CuTensorExecutorCache& executor_cache,
    const void* d_grouped_input,
    void* d_grouped_output,
    Index batch_size,
    cudaStream_t consumer_stream,
    cudaEvent_t input_ready_event,
    cudaEvent_t output_ready_event,
    const std::string& cache_prefix)
{
    const std::size_t grouped_bytes =
        batch_size * grouped_layout.grouped_size * sizeof(Complex);

    const CuTensorContractionDesc left_desc =
        make_cutensor_left_contraction_desc(
            target_sites,
            solver.model.local_dims,
            batch_size);
    const CuTensorContractionDesc right_desc =
        make_cutensor_right_contraction_desc(
            target_sites,
            solver.model.local_dims,
            batch_size);

    CuTensorExecutor* left_executor = nullptr;
    CuTensorExecutor* right_executor = nullptr;
    ExecutorInputBinding left_input_binding{};
    ExecutorInputBinding right_input_binding{};

    PetscErrorCode ierr = get_or_prepare_executor(
        executor_cache,
        cache_prefix + "_comm_left_apply_" + term_label + "_batch_" + std::to_string(batch_size),
        left_desc,
        cache_prefix + "_comm_left_operator_" + term_label,
        local_op,
        grouped_bytes,
        left_executor);
    if (ierr != 0) {
        return ierr;
    }

    ierr = get_or_prepare_executor(
        executor_cache,
        cache_prefix + "_comm_right_apply_" + term_label + "_batch_" + std::to_string(batch_size),
        right_desc,
        cache_prefix + "_comm_right_operator_" + term_label,
        local_op,
        grouped_bytes,
        right_executor);
    if (ierr != 0) {
        return ierr;
    }

    if (!bind_device_buffer_as_executor_input(
            d_grouped_input,
            grouped_bytes,
            input_ready_event,
            consumer_stream,
            *left_executor,
            left_input_binding) ||
        !bind_device_buffer_as_executor_input(
            d_grouped_input,
            grouped_bytes,
            input_ready_event,
            consumer_stream,
            *right_executor,
            right_input_binding)) {
        restore_executor_input_binding(right_input_binding);
        restore_executor_input_binding(left_input_binding);
        return PETSC_ERR_LIB;
    }

    if (!zero_executor_output(*left_executor, grouped_bytes) ||
        !zero_executor_output(*right_executor, grouped_bytes)) {
        restore_executor_input_binding(right_input_binding);
        restore_executor_input_binding(left_input_binding);
        return PETSC_ERR_LIB;
    }

    if (!execute_cutensor_executor_device(*left_executor) ||
        !execute_cutensor_executor_device(*right_executor)) {
        restore_executor_input_binding(right_input_binding);
        restore_executor_input_binding(left_input_binding);
        return PETSC_ERR_LIB;
    }

    const bool output_wait_ok =
        output_ready_event == nullptr ||
        wait_for_cuda_event(output_ready_event, left_executor->stream);
    if (!output_wait_ok ||
        !wait_for_cutensor_executor_completion(*right_executor, left_executor->stream)) {
        restore_executor_input_binding(right_input_binding);
        restore_executor_input_binding(left_input_binding);
        return PETSC_ERR_LIB;
    }

    if (!launch_commutator_combine_kernel(
            left_executor->d_output,
            right_executor->d_output,
            d_grouped_output,
            batch_size * grouped_layout.grouped_size,
            left_executor->stream)) {
        restore_executor_input_binding(right_input_binding);
        restore_executor_input_binding(left_input_binding);
        return PETSC_ERR_LIB;
    }

    if (!record_cutensor_executor_completion(*left_executor) ||
        !wait_for_executor_event(*left_executor, consumer_stream)) {
        restore_executor_input_binding(right_input_binding);
        restore_executor_input_binding(left_input_binding);
        return PETSC_ERR_LIB;
    }

    restore_executor_input_binding(right_input_binding);
    restore_executor_input_binding(left_input_binding);
    return 0;
}

PetscErrorCode apply_grouped_dissipator_cuda_buffer_impl(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& local_op_dag,
    const std::vector<Complex>& local_op_dag_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    CuTensorExecutorCache& executor_cache,
    const void* d_grouped_input,
    void* d_grouped_output,
    Index batch_size,
    cudaStream_t consumer_stream,
    cudaEvent_t input_ready_event,
    cudaEvent_t output_ready_event,
    const std::string& cache_prefix)
{
    const std::size_t grouped_bytes =
        batch_size * grouped_layout.grouped_size * sizeof(Complex);

    const CuTensorContractionDesc left_desc =
        make_cutensor_left_contraction_desc(
            target_sites,
            solver.model.local_dims,
            batch_size);
    const CuTensorContractionDesc right_desc =
        make_cutensor_right_contraction_desc(
            target_sites,
            solver.model.local_dims,
            batch_size);

    CuTensorExecutor* jump_left_executor = nullptr;
    CuTensorExecutor* jump_right_executor = nullptr;
    CuTensorExecutor* norm_left_executor = nullptr;
    CuTensorExecutor* norm_right_executor = nullptr;
    ExecutorInputBinding jump_left_input_binding{};
    ExecutorInputBinding jump_right_input_binding{};
    ExecutorInputBinding norm_left_input_binding{};
    ExecutorInputBinding norm_right_input_binding{};

    PetscErrorCode ierr = get_or_prepare_executor(
        executor_cache,
        cache_prefix + "_diss_jump_left_" + term_label + "_batch_" + std::to_string(batch_size),
        left_desc,
        cache_prefix + "_diss_L_" + term_label,
        local_op,
        grouped_bytes,
        jump_left_executor);
    if (ierr != 0) {
        return ierr;
    }

    ierr = get_or_prepare_executor(
        executor_cache,
        cache_prefix + "_diss_jump_right_" + term_label + "_batch_" + std::to_string(batch_size),
        right_desc,
        cache_prefix + "_diss_Ldag_" + term_label,
        local_op_dag,
        grouped_bytes,
        jump_right_executor);
    if (ierr != 0) {
        return ierr;
    }

    ierr = get_or_prepare_executor(
        executor_cache,
        cache_prefix + "_diss_norm_left_" + term_label + "_batch_" + std::to_string(batch_size),
        left_desc,
        cache_prefix + "_diss_LdagL_" + term_label,
        local_op_dag_op,
        grouped_bytes,
        norm_left_executor);
    if (ierr != 0) {
        return ierr;
    }

    ierr = get_or_prepare_executor(
        executor_cache,
        cache_prefix + "_diss_norm_right_" + term_label + "_batch_" + std::to_string(batch_size),
        right_desc,
        cache_prefix + "_diss_LdagL_" + term_label,
        local_op_dag_op,
        grouped_bytes,
        norm_right_executor);
    if (ierr != 0) {
        return ierr;
    }

    if (!bind_device_buffer_as_executor_input(
            d_grouped_input,
            grouped_bytes,
            input_ready_event,
            consumer_stream,
            *jump_left_executor,
            jump_left_input_binding) ||
        !bind_device_buffer_as_executor_input(
            d_grouped_input,
            grouped_bytes,
            input_ready_event,
            consumer_stream,
            *norm_left_executor,
            norm_left_input_binding) ||
        !bind_device_buffer_as_executor_input(
            d_grouped_input,
            grouped_bytes,
            input_ready_event,
            consumer_stream,
            *norm_right_executor,
            norm_right_input_binding)) {
        restore_executor_input_binding(norm_right_input_binding);
        restore_executor_input_binding(norm_left_input_binding);
        restore_executor_input_binding(jump_right_input_binding);
        restore_executor_input_binding(jump_left_input_binding);
        return PETSC_ERR_LIB;
    }

    if (!zero_executor_output(*jump_left_executor, grouped_bytes) ||
        !zero_executor_output(*jump_right_executor, grouped_bytes) ||
        !zero_executor_output(*norm_left_executor, grouped_bytes) ||
        !zero_executor_output(*norm_right_executor, grouped_bytes)) {
        restore_executor_input_binding(norm_right_input_binding);
        restore_executor_input_binding(norm_left_input_binding);
        restore_executor_input_binding(jump_right_input_binding);
        restore_executor_input_binding(jump_left_input_binding);
        return PETSC_ERR_LIB;
    }

    if (!execute_cutensor_executor_device(*jump_left_executor)) {
        restore_executor_input_binding(norm_right_input_binding);
        restore_executor_input_binding(norm_left_input_binding);
        restore_executor_input_binding(jump_right_input_binding);
        restore_executor_input_binding(jump_left_input_binding);
        return PETSC_ERR_LIB;
    }

    if (!bind_executor_output_as_executor_input(
            *jump_left_executor,
            *jump_right_executor,
            jump_right_input_binding) ||
        !execute_cutensor_executor_device(*jump_right_executor)) {
        restore_executor_input_binding(norm_right_input_binding);
        restore_executor_input_binding(norm_left_input_binding);
        restore_executor_input_binding(jump_right_input_binding);
        restore_executor_input_binding(jump_left_input_binding);
        return PETSC_ERR_LIB;
    }

    if (!execute_cutensor_executor_device(*norm_left_executor) ||
        !execute_cutensor_executor_device(*norm_right_executor)) {
        restore_executor_input_binding(norm_right_input_binding);
        restore_executor_input_binding(norm_left_input_binding);
        restore_executor_input_binding(jump_right_input_binding);
        restore_executor_input_binding(jump_left_input_binding);
        return PETSC_ERR_LIB;
    }

    const bool output_wait_ok =
        output_ready_event == nullptr ||
        wait_for_cuda_event(output_ready_event, norm_right_executor->stream);
    if (!output_wait_ok ||
        !wait_for_cutensor_executor_completion(*jump_right_executor, norm_right_executor->stream) ||
        !wait_for_cutensor_executor_completion(*norm_left_executor, norm_right_executor->stream)) {
        restore_executor_input_binding(norm_right_input_binding);
        restore_executor_input_binding(norm_left_input_binding);
        restore_executor_input_binding(jump_right_input_binding);
        restore_executor_input_binding(jump_left_input_binding);
        return PETSC_ERR_LIB;
    }

    if (!launch_dissipator_combine_kernel(
            jump_right_executor->d_output,
            norm_left_executor->d_output,
            norm_right_executor->d_output,
            d_grouped_output,
            batch_size * grouped_layout.grouped_size,
            norm_right_executor->stream)) {
        restore_executor_input_binding(norm_right_input_binding);
        restore_executor_input_binding(norm_left_input_binding);
        restore_executor_input_binding(jump_right_input_binding);
        restore_executor_input_binding(jump_left_input_binding);
        return PETSC_ERR_LIB;
    }

    if (!record_cutensor_executor_completion(*norm_right_executor) ||
        !wait_for_executor_event(*norm_right_executor, consumer_stream)) {
        restore_executor_input_binding(norm_right_input_binding);
        restore_executor_input_binding(norm_left_input_binding);
        restore_executor_input_binding(jump_right_input_binding);
        restore_executor_input_binding(jump_left_input_binding);
        return PETSC_ERR_LIB;
    }

    restore_executor_input_binding(norm_right_input_binding);
    restore_executor_input_binding(norm_left_input_binding);
    restore_executor_input_binding(jump_right_input_binding);
    restore_executor_input_binding(jump_left_input_binding);
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
    Index batch_size,
    cudaStream_t consumer_stream)
{
    const CuTensorContractionDesc left_desc =
        make_cutensor_left_contraction_desc(
            target_sites,
            solver.model.local_dims,
            batch_size);

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
        batch_size,
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
    Index batch_size,
    cudaStream_t consumer_stream)
{
    const CuTensorContractionDesc right_desc =
        make_cutensor_right_contraction_desc(
            target_sites,
            solver.model.local_dims,
            batch_size);

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
        batch_size,
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
    Index batch_size,
    cudaStream_t consumer_stream)
{
    const void* d_flat_input = nullptr;
    void* d_flat_output = nullptr;

    PetscCall(get_petsc_vec_device_read_ptr(x, d_flat_input));
    PetscCall(get_petsc_vec_device_write_ptr(y, d_flat_output));

    CuTensorExecutor* regroup_executor = nullptr;
    const CuTensorContractionDesc left_desc =
        make_cutensor_left_contraction_desc(
            target_sites,
            solver.model.local_dims,
            batch_size);
    const std::size_t grouped_bytes =
        batch_size * grouped_layout.grouped_size * sizeof(Complex);

    PetscErrorCode ierr = get_or_prepare_executor(
        executor_cache,
        "petsc_grouped_regroup_buffer_" + term_label + "_batch_" + std::to_string(batch_size),
        left_desc,
        "petsc_grouped_regroup_operator_" + term_label,
        local_op,
        grouped_bytes,
        regroup_executor);
    if (ierr != 0) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return ierr;
    }

    const bool regroup_ok =
        launch_flat_to_grouped_batched_kernel(
            cuda_grouped_layout,
            batch_size,
            d_flat_input,
            regroup_executor->d_input,
            regroup_executor->stream);

    if (!regroup_ok) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    if (!record_cutensor_executor_completion(*regroup_executor)) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    ierr = apply_grouped_commutator_cuda_buffer_impl(
        solver,
        term_label,
        local_op,
        target_sites,
        grouped_layout,
        executor_cache,
        regroup_executor->d_input,
        regroup_executor->d_output,
        batch_size,
        consumer_stream,
        regroup_executor->completion_event,
        nullptr,
        "petsc_grouped");
    if (ierr != 0) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return ierr;
    }

    ierr = flatten_grouped_buffer_to_device_ptr(
        cuda_grouped_layout,
        regroup_executor->d_output,
        d_flat_output,
        batch_size,
        consumer_stream,
        regroup_executor->stream);
    if (ierr != 0) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return ierr;
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
    Index batch_size,
    cudaStream_t consumer_stream)
{
    const void* d_flat_input = nullptr;
    void* d_flat_output = nullptr;

    PetscCall(get_petsc_vec_device_read_ptr(x, d_flat_input));
    PetscCall(get_petsc_vec_device_write_ptr(y, d_flat_output));

    CuTensorExecutor* regroup_executor = nullptr;
    const CuTensorContractionDesc left_desc =
        make_cutensor_left_contraction_desc(
            target_sites,
            solver.model.local_dims,
            batch_size);
    const std::size_t grouped_bytes =
        batch_size * grouped_layout.grouped_size * sizeof(Complex);

    PetscErrorCode ierr = get_or_prepare_executor(
        executor_cache,
        "petsc_grouped_diss_regroup_buffer_" + term_label + "_batch_" + std::to_string(batch_size),
        left_desc,
        "petsc_grouped_diss_regroup_operator_" + term_label,
        local_op,
        grouped_bytes,
        regroup_executor);
    if (ierr != 0) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return ierr;
    }

    const bool regroup_ok =
        launch_flat_to_grouped_batched_kernel(
            cuda_grouped_layout,
            batch_size,
            d_flat_input,
            regroup_executor->d_input,
            regroup_executor->stream);

    if (!regroup_ok) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    if (!record_cutensor_executor_completion(*regroup_executor)) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return PETSC_ERR_LIB;
    }

    ierr = apply_grouped_dissipator_cuda_buffer_impl(
        solver,
        term_label,
        local_op,
        local_op_dag,
        local_op_dag_op,
        target_sites,
        grouped_layout,
        executor_cache,
        regroup_executor->d_input,
        regroup_executor->d_output,
        batch_size,
        consumer_stream,
        regroup_executor->completion_event,
        nullptr,
        "petsc_grouped");
    if (ierr != 0) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return ierr;
    }

    ierr = flatten_grouped_buffer_to_device_ptr(
        cuda_grouped_layout,
        regroup_executor->d_output,
        d_flat_output,
        batch_size,
        consumer_stream,
        regroup_executor->stream);
    if (ierr != 0) {
        PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
        PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
        return ierr;
    }

    PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
    PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
    return 0;
}

PetscErrorCode regroup_petsc_cuda_vec_to_grouped_buffer(
    const CudaGroupedStateLayout& cuda_grouped_layout,
    Vec x,
    void* d_grouped_output,
    Index batch_size,
    cudaStream_t stream)
{
    const void* d_flat_input = nullptr;
    PetscCall(get_petsc_vec_device_read_ptr(x, d_flat_input));

    const bool regroup_ok =
        launch_flat_to_grouped_batched_kernel(
            cuda_grouped_layout,
            batch_size,
            d_flat_input,
            d_grouped_output,
            stream);

    PetscCall(restore_petsc_vec_device_read_ptr(x, d_flat_input));
    return regroup_ok ? 0 : PETSC_ERR_LIB;
}

PetscErrorCode flatten_grouped_buffer_to_petsc_cuda_vec(
    const CudaGroupedStateLayout& cuda_grouped_layout,
    const void* d_grouped_input,
    Vec y,
    Index batch_size,
    cudaStream_t consumer_stream,
    cudaStream_t producer_stream)
{
    void* d_flat_output = nullptr;
    PetscCall(get_petsc_vec_device_write_ptr(y, d_flat_output));
    PetscErrorCode ierr = flatten_grouped_buffer_to_device_ptr(
        cuda_grouped_layout,
        d_grouped_input,
        d_flat_output,
        batch_size,
        consumer_stream,
        producer_stream);
    PetscCall(restore_petsc_vec_device_write_ptr(y, d_flat_output));
    return ierr;
}

PetscErrorCode apply_grouped_commutator_cuda_buffer(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    CuTensorExecutorCache& executor_cache,
    const void* d_grouped_input,
    void* d_grouped_output,
    Index batch_size,
    cudaStream_t consumer_stream,
    cudaEvent_t input_ready_event,
    cudaEvent_t output_ready_event)
{
    return apply_grouped_commutator_cuda_buffer_impl(
        solver,
        term_label,
        local_op,
        target_sites,
        grouped_layout,
        executor_cache,
        d_grouped_input,
        d_grouped_output,
        batch_size,
        consumer_stream,
        input_ready_event,
        output_ready_event,
        "petsc_grouped_buffer");
}

PetscErrorCode apply_grouped_dissipator_cuda_buffer(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& local_op_dag,
    const std::vector<Complex>& local_op_dag_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    CuTensorExecutorCache& executor_cache,
    const void* d_grouped_input,
    void* d_grouped_output,
    Index batch_size,
    cudaStream_t consumer_stream,
    cudaEvent_t input_ready_event,
    cudaEvent_t output_ready_event)
{
    return apply_grouped_dissipator_cuda_buffer_impl(
        solver,
        term_label,
        local_op,
        local_op_dag,
        local_op_dag_op,
        target_sites,
        grouped_layout,
        executor_cache,
        d_grouped_input,
        d_grouped_output,
        batch_size,
        consumer_stream,
        input_ready_event,
        output_ready_event,
        "petsc_grouped_buffer");
}

} // namespace culindblad

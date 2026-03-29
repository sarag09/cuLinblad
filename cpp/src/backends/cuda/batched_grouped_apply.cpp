#include "culindblad/batched_grouped_apply.hpp"

#include <chrono>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "culindblad/batched_grouped_layout.hpp"
#include "culindblad/cuda_grouped_layout.hpp"
#include "culindblad/cutensor_contraction_desc.hpp"
#include "culindblad/cutensor_executor.hpp"
#include "culindblad/grouped_state_layout.hpp"
#include "culindblad/cuda_elementwise.hpp"

namespace culindblad {

std::vector<std::vector<Complex>> apply_batched_grouped_left_cuda_prototype(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<std::vector<Complex>>& flat_states)
{
    if (flat_states.empty()) {
        return {};
    }

    GroupedStateLayout single_layout =
        make_grouped_state_layout(target_sites, solver.model.local_dims);

    BatchedGroupedLayout batched_layout =
        make_batched_grouped_layout(single_layout, flat_states.size());

    std::vector<Complex> grouped_batch;
    pack_flat_batch_to_grouped_batch(
        batched_layout,
        flat_states,
        grouped_batch);

    CudaGroupedStateLayout cuda_grouped_layout{};
    if (!create_cuda_grouped_state_layout(single_layout, cuda_grouped_layout)) {
        throw std::runtime_error(
            "apply_batched_grouped_left_cuda_prototype: failed to create CUDA grouped layout");
    }

    const CuTensorContractionDesc left_desc =
        make_cutensor_left_contraction_desc(target_sites, solver.model.local_dims);

    CuTensorExecutor executor{};
    const std::size_t grouped_bytes =
        single_layout.grouped_size * sizeof(Complex);

    if (!create_cutensor_executor(
            left_desc,
            local_op.size() * sizeof(Complex),
            grouped_bytes,
            grouped_bytes,
            executor)) {
        (void)destroy_cuda_grouped_state_layout(cuda_grouped_layout);
        throw std::runtime_error(
            "apply_batched_grouped_left_cuda_prototype: create_cutensor_executor failed");
    }

    if (!upload_cutensor_executor_operator(executor, local_op)) {
        (void)destroy_cutensor_executor(executor);
        (void)destroy_cuda_grouped_state_layout(cuda_grouped_layout);
        throw std::runtime_error(
            "apply_batched_grouped_left_cuda_prototype: operator upload failed");
    }

    std::vector<Complex> grouped_output_batch(
        batched_layout.total_grouped_size,
        Complex{0.0, 0.0});

    for (Index batch = 0; batch < batched_layout.batch_size; ++batch) {
        const Index offset = batch * batched_layout.per_state_grouped_size;

        std::vector<Complex> grouped_single_in(
            grouped_batch.begin() + offset,
            grouped_batch.begin() + offset + batched_layout.per_state_grouped_size);

        std::vector<Complex> grouped_single_out(
            batched_layout.per_state_grouped_size,
            Complex{0.0, 0.0});

        if (!execute_cutensor_executor_with_resident_operator(
                executor,
                grouped_single_in,
                grouped_single_out)) {
            (void)destroy_cutensor_executor(executor);
            (void)destroy_cuda_grouped_state_layout(cuda_grouped_layout);
            throw std::runtime_error(
                "apply_batched_grouped_left_cuda_prototype: executor run failed");
        }

        for (Index i = 0; i < batched_layout.per_state_grouped_size; ++i) {
            grouped_output_batch[offset + i] = grouped_single_out[i];
        }
    }

    std::vector<std::vector<Complex>> flat_output_states;
    unpack_grouped_batch_to_flat_batch(
        batched_layout,
        grouped_output_batch,
        flat_output_states);

    (void)destroy_cutensor_executor(executor);
    (void)destroy_cuda_grouped_state_layout(cuda_grouped_layout);

    return flat_output_states;
}

std::vector<std::vector<Complex>> apply_batched_grouped_left_cuda_device_staged(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<std::vector<Complex>>& flat_states)
{
    if (flat_states.empty()) {
        return {};
    }

    GroupedStateLayout single_layout =
        make_grouped_state_layout(target_sites, solver.model.local_dims);

    BatchedGroupedLayout batched_layout =
        make_batched_grouped_layout(single_layout, flat_states.size());

    std::vector<Complex> grouped_batch;
    pack_flat_batch_to_grouped_batch(
        batched_layout,
        flat_states,
        grouped_batch);

    CudaGroupedStateLayout cuda_grouped_layout{};
    if (!create_cuda_grouped_state_layout(single_layout, cuda_grouped_layout)) {
        throw std::runtime_error(
            "apply_batched_grouped_left_cuda_device_staged: failed to create CUDA grouped layout");
    }

    const CuTensorContractionDesc left_desc =
        make_cutensor_left_contraction_desc(target_sites, solver.model.local_dims);

    CuTensorExecutor executor{};
    const std::size_t grouped_bytes =
        single_layout.grouped_size * sizeof(Complex);

    if (!create_cutensor_executor(
            left_desc,
            local_op.size() * sizeof(Complex),
            grouped_bytes,
            grouped_bytes,
            executor)) {
        (void)destroy_cuda_grouped_state_layout(cuda_grouped_layout);
        throw std::runtime_error(
            "apply_batched_grouped_left_cuda_device_staged: create_cutensor_executor failed");
    }

    if (!upload_cutensor_executor_operator(executor, local_op)) {
        (void)destroy_cutensor_executor(executor);
        (void)destroy_cuda_grouped_state_layout(cuda_grouped_layout);
        throw std::runtime_error(
            "apply_batched_grouped_left_cuda_device_staged: operator upload failed");
    }

    void* d_batched_input = nullptr;
    void* d_batched_output = nullptr;

    const std::size_t total_grouped_bytes =
        batched_layout.total_grouped_size * sizeof(Complex);

    if (cudaMalloc(&d_batched_input, total_grouped_bytes) != cudaSuccess ||
        cudaMalloc(&d_batched_output, total_grouped_bytes) != cudaSuccess) {
        if (d_batched_output != nullptr) {
            cudaFree(d_batched_output);
        }
        if (d_batched_input != nullptr) {
            cudaFree(d_batched_input);
        }
        (void)destroy_cutensor_executor(executor);
        (void)destroy_cuda_grouped_state_layout(cuda_grouped_layout);
        throw std::runtime_error(
            "apply_batched_grouped_left_cuda_device_staged: batched device allocation failed");
    }

    if (cudaMemcpy(
            d_batched_input,
            grouped_batch.data(),
            total_grouped_bytes,
            cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_batched_output);
        cudaFree(d_batched_input);
        (void)destroy_cutensor_executor(executor);
        (void)destroy_cuda_grouped_state_layout(cuda_grouped_layout);
        throw std::runtime_error(
            "apply_batched_grouped_left_cuda_device_staged: batched input upload failed");
    }

    if (!launch_zero_batched_buffer_kernel(
            d_batched_output,
            batched_layout.total_grouped_size,
            executor.stream)) {
        cudaFree(d_batched_output);
        cudaFree(d_batched_input);
        (void)destroy_cutensor_executor(executor);
        (void)destroy_cuda_grouped_state_layout(cuda_grouped_layout);
        throw std::runtime_error(
            "apply_batched_grouped_left_cuda_device_staged: batched output zero kernel failed");
    }

    if (cudaStreamSynchronize(executor.stream) != cudaSuccess) {
        cudaFree(d_batched_output);
        cudaFree(d_batched_input);
        (void)destroy_cutensor_executor(executor);
        (void)destroy_cuda_grouped_state_layout(cuda_grouped_layout);
        throw std::runtime_error(
            "apply_batched_grouped_left_cuda_device_staged: batched output zero synchronize failed");
    }

    const void* saved_d_input = executor.d_input;
    void* saved_d_output = executor.d_output;

    for (Index batch = 0; batch < batched_layout.batch_size; ++batch) {
        const std::size_t byte_offset = batch * grouped_bytes;

        executor.d_input =
            static_cast<void*>(static_cast<char*>(d_batched_input) + byte_offset);
        executor.d_output =
            static_cast<void*>(static_cast<char*>(d_batched_output) + byte_offset);

        if (cudaMemsetAsync(
                executor.d_output,
                0,
                grouped_bytes,
                executor.stream) != cudaSuccess) {
            executor.d_input = const_cast<void*>(saved_d_input);
            executor.d_output = saved_d_output;
            cudaFree(d_batched_output);
            cudaFree(d_batched_input);
            (void)destroy_cutensor_executor(executor);
            (void)destroy_cuda_grouped_state_layout(cuda_grouped_layout);
            throw std::runtime_error(
                "apply_batched_grouped_left_cuda_device_staged: per-state output memset failed");
        }

        if (!execute_cutensor_executor_device(executor)) {
            executor.d_input = const_cast<void*>(saved_d_input);
            executor.d_output = saved_d_output;
            cudaFree(d_batched_output);
            cudaFree(d_batched_input);
            (void)destroy_cutensor_executor(executor);
            (void)destroy_cuda_grouped_state_layout(cuda_grouped_layout);
            throw std::runtime_error(
                "apply_batched_grouped_left_cuda_device_staged: executor run failed");
        }
    }

    executor.d_input = const_cast<void*>(saved_d_input);
    executor.d_output = saved_d_output;

    if (cudaStreamSynchronize(executor.stream) != cudaSuccess) {
        cudaFree(d_batched_output);
        cudaFree(d_batched_input);
        (void)destroy_cutensor_executor(executor);
        (void)destroy_cuda_grouped_state_layout(cuda_grouped_layout);
        throw std::runtime_error(
            "apply_batched_grouped_left_cuda_device_staged: stream synchronize failed");
    }

    std::vector<Complex> grouped_output_batch(
        batched_layout.total_grouped_size,
        Complex{0.0, 0.0});

    if (cudaMemcpy(
            grouped_output_batch.data(),
            d_batched_output,
            total_grouped_bytes,
            cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_batched_output);
        cudaFree(d_batched_input);
        (void)destroy_cutensor_executor(executor);
        (void)destroy_cuda_grouped_state_layout(cuda_grouped_layout);
        throw std::runtime_error(
            "apply_batched_grouped_left_cuda_device_staged: batched output download failed");
    }

    std::vector<std::vector<Complex>> flat_output_states;
    unpack_grouped_batch_to_flat_batch(
        batched_layout,
        grouped_output_batch,
        flat_output_states);

    cudaFree(d_batched_output);
    cudaFree(d_batched_input);
    (void)destroy_cutensor_executor(executor);
    (void)destroy_cuda_grouped_state_layout(cuda_grouped_layout);

    return flat_output_states;
}

BatchedGroupedApplyTiming time_batched_grouped_left_cuda_prototype(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<std::vector<Complex>>& flat_states)
{
    const auto t_start = std::chrono::steady_clock::now();

    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;

    if (cudaEventCreate(&gpu_start) != cudaSuccess) {
        throw std::runtime_error(
            "time_batched_grouped_left_cuda_prototype: cudaEventCreate start failed");
    }

    if (cudaEventCreate(&gpu_stop) != cudaSuccess) {
        cudaEventDestroy(gpu_start);
        throw std::runtime_error(
            "time_batched_grouped_left_cuda_prototype: cudaEventCreate stop failed");
    }

    if (cudaEventRecord(gpu_start, 0) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error(
            "time_batched_grouped_left_cuda_prototype: cudaEventRecord start failed");
    }

    std::vector<std::vector<Complex>> output_states =
        apply_batched_grouped_left_cuda_prototype(
            solver,
            local_op,
            target_sites,
            flat_states);

    if (cudaEventRecord(gpu_stop, 0) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error(
            "time_batched_grouped_left_cuda_prototype: cudaEventRecord stop failed");
    }

    if (cudaEventSynchronize(gpu_stop) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error(
            "time_batched_grouped_left_cuda_prototype: cudaEventSynchronize failed");
    }

    float gpu_milliseconds = 0.0f;
    if (cudaEventElapsedTime(&gpu_milliseconds, gpu_start, gpu_stop) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error(
            "time_batched_grouped_left_cuda_prototype: cudaEventElapsedTime failed");
    }

    cudaEventDestroy(gpu_stop);
    cudaEventDestroy(gpu_start);

    const auto t_end = std::chrono::steady_clock::now();

    BatchedGroupedApplyTiming timing;
    timing.wall_seconds =
        std::chrono::duration<double>(t_end - t_start).count();
    timing.gpu_seconds =
        static_cast<double>(gpu_milliseconds) * 1.0e-3;
    timing.output_states = std::move(output_states);
    return timing;
}

BatchedGroupedApplyTiming time_batched_grouped_left_cuda_device_staged(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<std::vector<Complex>>& flat_states)
{
    const auto t_start = std::chrono::steady_clock::now();

    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;

    if (cudaEventCreate(&gpu_start) != cudaSuccess) {
        throw std::runtime_error(
            "time_batched_grouped_left_cuda_device_staged: cudaEventCreate start failed");
    }

    if (cudaEventCreate(&gpu_stop) != cudaSuccess) {
        cudaEventDestroy(gpu_start);
        throw std::runtime_error(
            "time_batched_grouped_left_cuda_device_staged: cudaEventCreate stop failed");
    }

    if (cudaEventRecord(gpu_start, 0) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error(
            "time_batched_grouped_left_cuda_device_staged: cudaEventRecord start failed");
    }

    std::vector<std::vector<Complex>> output_states =
        apply_batched_grouped_left_cuda_device_staged(
            solver,
            local_op,
            target_sites,
            flat_states);

    if (cudaEventRecord(gpu_stop, 0) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error(
            "time_batched_grouped_left_cuda_device_staged: cudaEventRecord stop failed");
    }

    if (cudaEventSynchronize(gpu_stop) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error(
            "time_batched_grouped_left_cuda_device_staged: cudaEventSynchronize failed");
    }

    float gpu_milliseconds = 0.0f;
    if (cudaEventElapsedTime(&gpu_milliseconds, gpu_start, gpu_stop) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error(
            "time_batched_grouped_left_cuda_device_staged: cudaEventElapsedTime failed");
    }

    cudaEventDestroy(gpu_stop);
    cudaEventDestroy(gpu_start);

    const auto t_end = std::chrono::steady_clock::now();

    BatchedGroupedApplyTiming timing;
    timing.wall_seconds =
        std::chrono::duration<double>(t_end - t_start).count();
    timing.gpu_seconds =
        static_cast<double>(gpu_milliseconds) * 1.0e-3;
    timing.output_states = std::move(output_states);
    return timing;
}

} // namespace culindblad
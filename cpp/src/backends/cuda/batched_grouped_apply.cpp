#include "culindblad/batched_grouped_apply.hpp"

#include <chrono>
#include <stdexcept>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>

#include "culindblad/batched_grouped_layout.hpp"
#include "culindblad/cuda_grouped_layout.hpp"
#include "culindblad/cutensor_contraction_desc.hpp"
#include "culindblad/cutensor_executor.hpp"
#include "culindblad/grouped_state_layout.hpp"
#include "culindblad/cuda_elementwise.hpp"

namespace culindblad {

namespace {

BatchedGroupedApplyTiming make_timing_result(
    const std::chrono::steady_clock::time_point& t_start,
    const std::chrono::steady_clock::time_point& t_end,
    float gpu_milliseconds,
    std::vector<std::vector<Complex>> output_states)
{
    BatchedGroupedApplyTiming timing;
    timing.wall_seconds =
        std::chrono::duration<double>(t_end - t_start).count();
    timing.gpu_seconds =
        static_cast<double>(gpu_milliseconds) * 1.0e-3;
    timing.output_states = std::move(output_states);
    return timing;
}

BatchedGroupedApplyTiming time_apply_helper(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<std::vector<Complex>>& flat_states,
    std::vector<std::vector<Complex>> (*apply_fn)(
        const Solver&,
        const std::vector<Complex>&,
        const std::vector<Index>&,
        const std::vector<std::vector<Complex>>&))
{
    const auto t_start = std::chrono::steady_clock::now();

    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;

    if (cudaEventCreate(&gpu_start) != cudaSuccess) {
        throw std::runtime_error(
            "time_apply_helper: cudaEventCreate start failed");
    }

    if (cudaEventCreate(&gpu_stop) != cudaSuccess) {
        cudaEventDestroy(gpu_start);
        throw std::runtime_error(
            "time_apply_helper: cudaEventCreate stop failed");
    }

    if (cudaEventRecord(gpu_start, 0) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error(
            "time_apply_helper: cudaEventRecord start failed");
    }

    std::vector<std::vector<Complex>> output_states =
        apply_fn(
            solver,
            local_op,
            target_sites,
            flat_states);

    if (cudaEventRecord(gpu_stop, 0) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error(
            "time_apply_helper: cudaEventRecord stop failed");
    }

    if (cudaEventSynchronize(gpu_stop) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error(
            "time_apply_helper: cudaEventSynchronize failed");
    }

    float gpu_milliseconds = 0.0f;
    if (cudaEventElapsedTime(&gpu_milliseconds, gpu_start, gpu_stop) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error(
            "time_apply_helper: cudaEventElapsedTime failed");
    }

    cudaEventDestroy(gpu_stop);
    cudaEventDestroy(gpu_start);

    const auto t_end = std::chrono::steady_clock::now();

    return make_timing_result(
        t_start,
        t_end,
        gpu_milliseconds,
        std::move(output_states));
}

struct BatchedDeviceBuffers {
    void* d_input = nullptr;
    void* d_output = nullptr;
    std::size_t total_bytes = 0;
};

void allocate_batched_device_buffers(
    std::size_t total_grouped_bytes,
    BatchedDeviceBuffers& buffers)
{
    buffers.total_bytes = total_grouped_bytes;

    if (cudaMalloc(&buffers.d_input, total_grouped_bytes) != cudaSuccess ||
        cudaMalloc(&buffers.d_output, total_grouped_bytes) != cudaSuccess) {
        if (buffers.d_output != nullptr) {
            cudaFree(buffers.d_output);
            buffers.d_output = nullptr;
        }
        if (buffers.d_input != nullptr) {
            cudaFree(buffers.d_input);
            buffers.d_input = nullptr;
        }
        throw std::runtime_error(
            "allocate_batched_device_buffers: device allocation failed");
    }
}

void free_batched_device_buffers(
    BatchedDeviceBuffers& buffers)
{
    if (buffers.d_output != nullptr) {
        cudaFree(buffers.d_output);
        buffers.d_output = nullptr;
    }
    if (buffers.d_input != nullptr) {
        cudaFree(buffers.d_input);
        buffers.d_input = nullptr;
    }
    buffers.total_bytes = 0;
}

void upload_batched_grouped_input(
    const std::vector<Complex>& grouped_batch,
    BatchedDeviceBuffers& buffers)
{
    if (cudaMemcpy(
            buffers.d_input,
            grouped_batch.data(),
            buffers.total_bytes,
            cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error(
            "upload_batched_grouped_input: batched input upload failed");
    }
}

void zero_batched_grouped_output(
    CuTensorExecutor& executor,
    BatchedDeviceBuffers& buffers)
{
    if (cudaMemsetAsync(
            buffers.d_output,
            0,
            buffers.total_bytes,
            executor.stream) != cudaSuccess) {
        throw std::runtime_error(
            "zero_batched_grouped_output: batched output memset failed");
    }
}

void execute_batched_grouped_left_serial_subranges(
    CuTensorExecutor& executor,
    const BatchedGroupedLayout& batched_layout,
    BatchedDeviceBuffers& buffers)
{
    const std::size_t grouped_bytes =
        batched_layout.per_state_grouped_size * sizeof(Complex);

    const void* saved_d_input = executor.d_input;
    void* saved_d_output = executor.d_output;

    for (Index batch = 0; batch < batched_layout.batch_size; ++batch) {
        const std::size_t byte_offset = batch * grouped_bytes;

        executor.d_input =
            static_cast<void*>(static_cast<char*>(buffers.d_input) + byte_offset);
        executor.d_output =
            static_cast<void*>(static_cast<char*>(buffers.d_output) + byte_offset);

        if (cudaMemsetAsync(
                executor.d_output,
                0,
                grouped_bytes,
                executor.stream) != cudaSuccess) {
            executor.d_input = const_cast<void*>(saved_d_input);
            executor.d_output = saved_d_output;
            throw std::runtime_error(
                "execute_batched_grouped_left_serial_subranges: per-state output memset failed");
        }

        if (!execute_cutensor_executor_device(executor)) {
            executor.d_input = const_cast<void*>(saved_d_input);
            executor.d_output = saved_d_output;
            throw std::runtime_error(
                "execute_batched_grouped_left_serial_subranges: executor run failed");
        }
    }

    executor.d_input = const_cast<void*>(saved_d_input);
    executor.d_output = saved_d_output;
}

std::vector<std::vector<Complex>> finalize_batched_grouped_output(
    const BatchedGroupedLayout& batched_layout,
    BatchedDeviceBuffers& buffers)
{
    std::vector<Complex> grouped_output_batch(
        batched_layout.total_grouped_size,
        Complex{0.0, 0.0});

    if (cudaMemcpy(
            grouped_output_batch.data(),
            buffers.d_output,
            buffers.total_bytes,
            cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error(
            "finalize_batched_grouped_output: batched output download failed");
    }

    std::vector<std::vector<Complex>> flat_output_states;
    unpack_grouped_batch_to_flat_batch(
        batched_layout,
        grouped_output_batch,
        flat_output_states);

    return flat_output_states;
}

std::vector<std::vector<Complex>> apply_batched_grouped_left_with_backend(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<std::vector<Complex>>& flat_states,
    bool use_device_staged_backend)
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
            "apply_batched_grouped_left_with_backend: failed to create CUDA grouped layout");
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
            "apply_batched_grouped_left_with_backend: create_cutensor_executor failed");
    }

    if (!upload_cutensor_executor_operator(executor, local_op)) {
        (void)destroy_cutensor_executor(executor);
        (void)destroy_cuda_grouped_state_layout(cuda_grouped_layout);
        throw std::runtime_error(
            "apply_batched_grouped_left_with_backend: operator upload failed");
    }

    try {
        if (!use_device_staged_backend) {
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
                    throw std::runtime_error(
                        "apply_batched_grouped_left_with_backend: prototype executor run failed");
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

        BatchedDeviceBuffers buffers{};
        allocate_batched_device_buffers(
            batched_layout.total_grouped_size * sizeof(Complex),
            buffers);

        try {
            upload_batched_grouped_input(grouped_batch, buffers);
            zero_batched_grouped_output(executor, buffers);
            execute_batched_grouped_left_serial_subranges(
                executor,
                batched_layout,
                buffers);

            if (cudaStreamSynchronize(executor.stream) != cudaSuccess) {
                throw std::runtime_error(
                    "apply_batched_grouped_left_with_backend: stream synchronize failed");
            }

            std::vector<std::vector<Complex>> flat_output_states =
                finalize_batched_grouped_output(
                    batched_layout,
                    buffers);

            free_batched_device_buffers(buffers);
            (void)destroy_cutensor_executor(executor);
            (void)destroy_cuda_grouped_state_layout(cuda_grouped_layout);
            return flat_output_states;
        } catch (...) {
            free_batched_device_buffers(buffers);
            throw;
        }
    } catch (...) {
        (void)destroy_cutensor_executor(executor);
        (void)destroy_cuda_grouped_state_layout(cuda_grouped_layout);
        throw;
    }
}

} // namespace

std::vector<std::vector<Complex>> apply_batched_grouped_left_cuda_prototype(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<std::vector<Complex>>& flat_states)
{
    return apply_batched_grouped_left_with_backend(
        solver,
        local_op,
        target_sites,
        flat_states,
        false);
}

std::vector<std::vector<Complex>> apply_batched_grouped_left_cuda_device_staged(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<std::vector<Complex>>& flat_states)
{
    return apply_batched_grouped_left_with_backend(
        solver,
        local_op,
        target_sites,
        flat_states,
        true);
}

std::vector<std::vector<Complex>> apply_batched_grouped_left_cuda_batch_object(
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

    const CuTensorContractionDesc left_desc =
        make_batched_cutensor_left_contraction_desc(
            target_sites,
            solver.model.local_dims,
            batched_layout.batch_size);

    CuTensorExecutor executor{};
    const std::size_t grouped_bytes =
        static_cast<std::size_t>(batched_layout.total_grouped_size) * sizeof(Complex);

    if (!create_cutensor_executor(
            left_desc,
            local_op.size() * sizeof(Complex),
            grouped_bytes,
            grouped_bytes,
            executor)) {
        throw std::runtime_error(
            "apply_batched_grouped_left_cuda_batch_object: create_cutensor_executor failed");
    }

    try {
        if (!upload_cutensor_executor_operator(executor, local_op)) {
            throw std::runtime_error(
                "apply_batched_grouped_left_cuda_batch_object: operator upload failed");
        }

        if (!upload_cutensor_executor_input(executor, grouped_batch)) {
            throw std::runtime_error(
                "apply_batched_grouped_left_cuda_batch_object: grouped batch upload failed");
        }

        if (!execute_cutensor_executor_device(executor)) {
            throw std::runtime_error(
                "apply_batched_grouped_left_cuda_batch_object: batched contraction failed");
        }

        std::vector<Complex> grouped_output_batch;
        if (!download_cutensor_executor_output(executor, grouped_output_batch)) {
            throw std::runtime_error(
                "apply_batched_grouped_left_cuda_batch_object: grouped batch download failed");
        }

        std::vector<std::vector<Complex>> flat_output_states;
        unpack_grouped_batch_to_flat_batch(
            batched_layout,
            grouped_output_batch,
            flat_output_states);

        (void)destroy_cutensor_executor(executor);
        return flat_output_states;
    } catch (...) {
        (void)destroy_cutensor_executor(executor);
        throw;
    }
}

std::vector<std::vector<Complex>> apply_batched_grouped_left_cuda_fused_candidate(
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

    Index target_hilbert_dim = 1;
    for (Index site : target_sites) {
        target_hilbert_dim *= solver.model.local_dims.at(site);
    }

    if (target_hilbert_dim == 0 || solver.layout.hilbert_dim % target_hilbert_dim != 0) {
        throw std::runtime_error(
            "apply_batched_grouped_left_cuda_fused_candidate: invalid target/complement dimensions");
    }

    const Index complement_dim =
        solver.layout.hilbert_dim / target_hilbert_dim;

    const Index op_dim =
        static_cast<Index>(std::llround(std::sqrt(static_cast<double>(local_op.size()))));

    if (op_dim != target_hilbert_dim ||
        local_op.size() != target_hilbert_dim * target_hilbert_dim) {
        throw std::runtime_error(
            "apply_batched_grouped_left_cuda_fused_candidate: local operator dimension mismatch");
    }

    std::vector<Complex> diagonal_op(target_hilbert_dim, Complex{0.0, 0.0});
    for (Index i = 0; i < target_hilbert_dim; ++i) {
        diagonal_op[i] = local_op[i * target_hilbert_dim + i];
    }

    void* d_diagonal_op = nullptr;
    void* d_grouped_input = nullptr;
    void* d_grouped_output = nullptr;

    const std::size_t diagonal_bytes =
        static_cast<std::size_t>(target_hilbert_dim) * sizeof(Complex);
    const std::size_t grouped_bytes =
        static_cast<std::size_t>(batched_layout.total_grouped_size) * sizeof(Complex);

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        throw std::runtime_error(
            "apply_batched_grouped_left_cuda_fused_candidate: stream creation failed");
    }

    try {
        if (cudaMalloc(&d_diagonal_op, diagonal_bytes) != cudaSuccess ||
            cudaMalloc(&d_grouped_input, grouped_bytes) != cudaSuccess ||
            cudaMalloc(&d_grouped_output, grouped_bytes) != cudaSuccess) {
            throw std::runtime_error(
                "apply_batched_grouped_left_cuda_fused_candidate: device allocation failed");
        }

        if (cudaMemcpyAsync(
                d_diagonal_op,
                diagonal_op.data(),
                diagonal_bytes,
                cudaMemcpyHostToDevice,
                stream) != cudaSuccess) {
            throw std::runtime_error(
                "apply_batched_grouped_left_cuda_fused_candidate: diagonal upload failed");
        }

        if (cudaMemcpyAsync(
                d_grouped_input,
                grouped_batch.data(),
                grouped_bytes,
                cudaMemcpyHostToDevice,
                stream) != cudaSuccess) {
            throw std::runtime_error(
                "apply_batched_grouped_left_cuda_fused_candidate: grouped input upload failed");
        }

        if (!launch_batched_grouped_left_diagonal_kernel(
                d_diagonal_op,
                d_grouped_input,
                d_grouped_output,
                target_hilbert_dim,
                complement_dim,
                batched_layout.batch_size,
                stream)) {
            throw std::runtime_error(
                "apply_batched_grouped_left_cuda_fused_candidate: batched diagonal kernel launch failed");
        }

        std::vector<Complex> grouped_output_batch(
            batched_layout.total_grouped_size,
            Complex{0.0, 0.0});

        if (cudaMemcpyAsync(
                grouped_output_batch.data(),
                d_grouped_output,
                grouped_bytes,
                cudaMemcpyDeviceToHost,
                stream) != cudaSuccess) {
            throw std::runtime_error(
                "apply_batched_grouped_left_cuda_fused_candidate: grouped output download failed");
        }

        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            throw std::runtime_error(
                "apply_batched_grouped_left_cuda_fused_candidate: stream synchronize failed");
        }

        std::vector<std::vector<Complex>> flat_output_states;
        unpack_grouped_batch_to_flat_batch(
            batched_layout,
            grouped_output_batch,
            flat_output_states);

        cudaFree(d_grouped_output);
        cudaFree(d_grouped_input);
        cudaFree(d_diagonal_op);
        cudaStreamDestroy(stream);

        return flat_output_states;
    } catch (...) {
        if (d_grouped_output != nullptr) {
            cudaFree(d_grouped_output);
        }
        if (d_grouped_input != nullptr) {
            cudaFree(d_grouped_input);
        }
        if (d_diagonal_op != nullptr) {
            cudaFree(d_diagonal_op);
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        throw;
    }
}

BatchedGroupedApplyTiming time_batched_grouped_left_cuda_prototype(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<std::vector<Complex>>& flat_states)
{
    return time_apply_helper(
        solver,
        local_op,
        target_sites,
        flat_states,
        apply_batched_grouped_left_cuda_prototype);
}

BatchedGroupedApplyTiming time_batched_grouped_left_cuda_device_staged(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<std::vector<Complex>>& flat_states)
{
    return time_apply_helper(
        solver,
        local_op,
        target_sites,
        flat_states,
        apply_batched_grouped_left_cuda_device_staged);
}

BatchedGroupedApplyTiming time_batched_grouped_left_cuda_batch_object(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<std::vector<Complex>>& flat_states)
{
    return time_apply_helper(
        solver,
        local_op,
        target_sites,
        flat_states,
        apply_batched_grouped_left_cuda_batch_object);
}

BatchedGroupedApplyTiming time_batched_grouped_left_cuda_fused_candidate(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<std::vector<Complex>>& flat_states)
{
    return time_apply_helper(
        solver,
        local_op,
        target_sites,
        flat_states,
        apply_batched_grouped_left_cuda_fused_candidate);
}

} // namespace culindblad

#include "culindblad/cuda_grouped_layout.hpp"

#include <cstddef>
#include <cuda_runtime.h>

#include "culindblad/types.hpp"

namespace culindblad {

namespace {

__device__ __forceinline__ std::size_t batch_major_offset(
    Index batch,
    Index per_batch_size,
    Index local_idx)
{
    return static_cast<std::size_t>(batch) * static_cast<std::size_t>(per_batch_size) +
           static_cast<std::size_t>(local_idx);
}

__global__ void flat_to_grouped_kernel(
    const Complex* flat_input,
    Complex* grouped_output,
    const Index* flat_to_grouped,
    Index flat_size)
{
    const Index idx =
        static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= flat_size) {
        return;
    }

    grouped_output[flat_to_grouped[idx]] = flat_input[idx];
}

__global__ void grouped_to_flat_kernel(
    const Complex* grouped_input,
    Complex* flat_output,
    const Index* grouped_to_flat,
    Index grouped_size)
{
    const Index idx =
        static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= grouped_size) {
        return;
    }

    flat_output[grouped_to_flat[idx]] = grouped_input[idx];
}

__global__ void flat_batch_to_grouped_kernel(
    const Complex* flat_input,
    Complex* grouped_output,
    const Index* flat_to_grouped,
    Index flat_size,
    Index grouped_size,
    Index batch_size)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total_size =
        static_cast<std::size_t>(flat_size) * static_cast<std::size_t>(batch_size);

    if (idx >= total_size) {
        return;
    }

    const Index batch =
        static_cast<Index>(idx / static_cast<std::size_t>(flat_size));
    const Index local_idx =
        static_cast<Index>(idx % static_cast<std::size_t>(flat_size));
    const Index grouped_idx =
        flat_to_grouped[local_idx];

    grouped_output[batch_major_offset(batch, grouped_size, grouped_idx)] =
        flat_input[batch_major_offset(batch, flat_size, local_idx)];
}

__global__ void grouped_batch_to_flat_kernel(
    const Complex* grouped_input,
    Complex* flat_output,
    const Index* grouped_to_flat,
    Index flat_size,
    Index grouped_size,
    Index batch_size)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total_size =
        static_cast<std::size_t>(grouped_size) * static_cast<std::size_t>(batch_size);

    if (idx >= total_size) {
        return;
    }

    const Index batch =
        static_cast<Index>(idx / static_cast<std::size_t>(grouped_size));
    const Index local_idx =
        static_cast<Index>(idx % static_cast<std::size_t>(grouped_size));
    const Index flat_idx =
        grouped_to_flat[local_idx];

    flat_output[batch_major_offset(batch, flat_size, flat_idx)] =
        grouped_input[batch_major_offset(batch, grouped_size, local_idx)];
}

bool cuda_malloc_indices(Index** ptr, std::size_t count)
{
    return cudaMalloc(reinterpret_cast<void**>(ptr), count * sizeof(Index)) == cudaSuccess;
}

bool cuda_copy_indices_to_device(Index* dst, const Index* src, std::size_t count)
{
    return cudaMemcpy(dst, src, count * sizeof(Index), cudaMemcpyHostToDevice) == cudaSuccess;
}

} // namespace

bool create_cuda_grouped_state_layout(
    const GroupedStateLayout& host_layout,
    CudaGroupedStateLayout& cuda_layout)
{
    cuda_layout.flat_size = host_layout.flat_density_to_grouped.size();
    cuda_layout.grouped_size = host_layout.grouped_to_flat_density.size();
    cuda_layout.d_flat_to_grouped = nullptr;
    cuda_layout.d_grouped_to_flat = nullptr;

    if (!cuda_malloc_indices(&cuda_layout.d_flat_to_grouped, cuda_layout.flat_size)) {
        return false;
    }

    if (!cuda_malloc_indices(&cuda_layout.d_grouped_to_flat, cuda_layout.grouped_size)) {
        cudaFree(cuda_layout.d_flat_to_grouped);
        cuda_layout.d_flat_to_grouped = nullptr;
        return false;
    }

    if (!cuda_copy_indices_to_device(
            cuda_layout.d_flat_to_grouped,
            host_layout.flat_density_to_grouped.data(),
            cuda_layout.flat_size)) {
        cudaFree(cuda_layout.d_grouped_to_flat);
        cudaFree(cuda_layout.d_flat_to_grouped);
        cuda_layout.d_grouped_to_flat = nullptr;
        cuda_layout.d_flat_to_grouped = nullptr;
        return false;
    }

    if (!cuda_copy_indices_to_device(
            cuda_layout.d_grouped_to_flat,
            host_layout.grouped_to_flat_density.data(),
            cuda_layout.grouped_size)) {
        cudaFree(cuda_layout.d_grouped_to_flat);
        cudaFree(cuda_layout.d_flat_to_grouped);
        cuda_layout.d_grouped_to_flat = nullptr;
        cuda_layout.d_flat_to_grouped = nullptr;
        return false;
    }

    return true;
}

bool destroy_cuda_grouped_state_layout(
    CudaGroupedStateLayout& cuda_layout)
{
    bool ok = true;

    if (cuda_layout.d_grouped_to_flat != nullptr &&
        cudaFree(cuda_layout.d_grouped_to_flat) != cudaSuccess) {
        ok = false;
    }

    if (cuda_layout.d_flat_to_grouped != nullptr &&
        cudaFree(cuda_layout.d_flat_to_grouped) != cudaSuccess) {
        ok = false;
    }

    cuda_layout.d_grouped_to_flat = nullptr;
    cuda_layout.d_flat_to_grouped = nullptr;
    cuda_layout.flat_size = 0;
    cuda_layout.grouped_size = 0;

    return ok;
}

bool launch_flat_to_grouped_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    const void* d_flat_input,
    void* d_grouped_output,
    cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size =
        static_cast<int>((cuda_layout.flat_size + block_size - 1) / block_size);

    flat_to_grouped_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const Complex*>(d_flat_input),
        reinterpret_cast<Complex*>(d_grouped_output),
        cuda_layout.d_flat_to_grouped,
        cuda_layout.flat_size);

    return cudaGetLastError() == cudaSuccess;
}

bool launch_grouped_to_flat_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    const void* d_grouped_input,
    void* d_flat_output,
    cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size =
        static_cast<int>((cuda_layout.grouped_size + block_size - 1) / block_size);

    grouped_to_flat_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const Complex*>(d_grouped_input),
        reinterpret_cast<Complex*>(d_flat_output),
        cuda_layout.d_grouped_to_flat,
        cuda_layout.grouped_size);

    return cudaGetLastError() == cudaSuccess;
}

bool launch_flat_batch_to_grouped_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    const void* d_flat_input,
    void* d_grouped_output,
    Index batch_size,
    cudaStream_t stream)
{
    const std::size_t total_size =
        static_cast<std::size_t>(cuda_layout.flat_size) * static_cast<std::size_t>(batch_size);
    const int block_size = 256;
    const int grid_size =
        static_cast<int>((total_size + block_size - 1) / block_size);

    flat_batch_to_grouped_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const Complex*>(d_flat_input),
        reinterpret_cast<Complex*>(d_grouped_output),
        cuda_layout.d_flat_to_grouped,
        cuda_layout.flat_size,
        cuda_layout.grouped_size,
        batch_size);

    return cudaGetLastError() == cudaSuccess;
}

bool launch_grouped_batch_to_flat_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    const void* d_grouped_input,
    void* d_flat_output,
    Index batch_size,
    cudaStream_t stream)
{
    const std::size_t total_size =
        static_cast<std::size_t>(cuda_layout.grouped_size) * static_cast<std::size_t>(batch_size);
    const int block_size = 256;
    const int grid_size =
        static_cast<int>((total_size + block_size - 1) / block_size);

    grouped_batch_to_flat_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const Complex*>(d_grouped_input),
        reinterpret_cast<Complex*>(d_flat_output),
        cuda_layout.d_grouped_to_flat,
        cuda_layout.flat_size,
        cuda_layout.grouped_size,
        batch_size);

    return cudaGetLastError() == cudaSuccess;
}

} // namespace culindblad

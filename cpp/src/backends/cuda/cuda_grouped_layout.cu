#include "culindblad/cuda_grouped_layout.hpp"

#include <cstddef>
#include <cuda_runtime.h>

#include "culindblad/types.hpp"

namespace culindblad {

namespace {

// Descriptor layout: each per-position triple is [full_hilbert_stride, local_dim,
// packed_weight] at offsets 3*p + {0,1,2}. (target_desc / complement_desc.)

// flat density index -> grouped index. flat = ket_full * D + bra_full; each of
// ket_full/bra_full is decomposed into target-site and complement-site sub-indices
// (matching make_grouped_state_layout's flatten_subset), then packed as
// ((ket_t*comp + ket_c)*target + bra_t)*comp + bra_c.
__device__ inline Index flat_to_grouped_index(
    Index flat, Index D, Index target_dim, Index complement_dim,
    const Index* tdesc, Index num_t, const Index* cdesc, Index num_c)
{
    Index ket_full = flat / D;
    Index bra_full = flat % D;
    Index ket_t = 0, ket_c = 0, bra_t = 0, bra_c = 0;
    for (Index p = 0; p < num_t; ++p) {
        const Index fs = tdesc[3 * p], dm = tdesc[3 * p + 1], w = tdesc[3 * p + 2];
        ket_t += ((ket_full / fs) % dm) * w;
        bra_t += ((bra_full / fs) % dm) * w;
    }
    for (Index q = 0; q < num_c; ++q) {
        const Index fs = cdesc[3 * q], dm = cdesc[3 * q + 1], w = cdesc[3 * q + 2];
        ket_c += ((ket_full / fs) % dm) * w;
        bra_c += ((bra_full / fs) % dm) * w;
    }
    return ((ket_t * complement_dim + ket_c) * target_dim + bra_t) * complement_dim + bra_c;
}

// grouped index -> flat density index (inverse of the above).
__device__ inline Index grouped_to_flat_index(
    Index g, Index D, Index target_dim, Index complement_dim,
    const Index* tdesc, Index num_t, const Index* cdesc, Index num_c)
{
    const Index bra_c = g % complement_dim;
    Index t = g / complement_dim;
    const Index bra_t = t % target_dim;
    t /= target_dim;
    const Index ket_c = t % complement_dim;
    const Index ket_t = t / complement_dim;
    Index ket_full = 0, bra_full = 0;
    for (Index p = 0; p < num_t; ++p) {
        const Index fs = tdesc[3 * p], dm = tdesc[3 * p + 1], w = tdesc[3 * p + 2];
        ket_full += ((ket_t / w) % dm) * fs;
        bra_full += ((bra_t / w) % dm) * fs;
    }
    for (Index q = 0; q < num_c; ++q) {
        const Index fs = cdesc[3 * q], dm = cdesc[3 * q + 1], w = cdesc[3 * q + 2];
        ket_full += ((ket_c / w) % dm) * fs;
        bra_full += ((bra_c / w) % dm) * fs;
    }
    return ket_full * D + bra_full;
}

__global__ void flat_to_grouped_kernel(
    const Complex* flat_input,
    Complex* grouped_output,
    Index flat_size,
    Index D, Index target_dim, Index complement_dim,
    const Index* tdesc, Index num_t, const Index* cdesc, Index num_c)
{
    const Index idx =
        static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= flat_size) {
        return;
    }
    const Index g = flat_to_grouped_index(idx, D, target_dim, complement_dim,
                                          tdesc, num_t, cdesc, num_c);
    grouped_output[g] = flat_input[idx];
}

__global__ void grouped_to_flat_kernel(
    const Complex* grouped_input,
    Complex* flat_output,
    Index grouped_size,
    Index D, Index target_dim, Index complement_dim,
    const Index* tdesc, Index num_t, const Index* cdesc, Index num_c)
{
    const Index idx =
        static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= grouped_size) {
        return;
    }
    const Index f = grouped_to_flat_index(idx, D, target_dim, complement_dim,
                                          tdesc, num_t, cdesc, num_c);
    flat_output[f] = grouped_input[idx];
}

__global__ void flat_to_grouped_batched_kernel(
    const Complex* flat_input,
    Complex* grouped_output,
    Index flat_size,
    Index grouped_size,
    Index total_flat_size,
    Index D, Index target_dim, Index complement_dim,
    const Index* tdesc, Index num_t, const Index* cdesc, Index num_c)
{
    const Index idx =
        static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x) +
        static_cast<Index>(threadIdx.x);
    if (idx >= total_flat_size) {
        return;
    }
    const Index batch = idx / flat_size;
    const Index local_flat = idx % flat_size;
    const Index g = flat_to_grouped_index(local_flat, D, target_dim, complement_dim,
                                          tdesc, num_t, cdesc, num_c);
    grouped_output[batch * grouped_size + g] = flat_input[idx];
}

__global__ void grouped_to_flat_batched_kernel(
    const Complex* grouped_input,
    Complex* flat_output,
    Index flat_size,
    Index grouped_size,
    Index total_grouped_size,
    Index D, Index target_dim, Index complement_dim,
    const Index* tdesc, Index num_t, const Index* cdesc, Index num_c)
{
    const Index idx =
        static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x) +
        static_cast<Index>(threadIdx.x);
    if (idx >= total_grouped_size) {
        return;
    }
    const Index batch = idx / grouped_size;
    const Index local_grouped = idx % grouped_size;
    const Index f = grouped_to_flat_index(local_grouped, D, target_dim, complement_dim,
                                          tdesc, num_t, cdesc, num_c);
    flat_output[batch * flat_size + f] = grouped_input[idx];
}

} // namespace

bool create_cuda_grouped_state_layout(
    const GroupedStateLayout& host_layout,
    CudaGroupedStateLayout& cuda_layout)
{
    const std::vector<Index>& dims = host_layout.local_dims;
    const std::vector<Index>& target_sites = host_layout.target_sites;
    const Index num_sites = dims.size();

    cuda_layout.flat_size = host_layout.hilbert_dim * host_layout.hilbert_dim;
    cuda_layout.grouped_size = host_layout.grouped_size;
    cuda_layout.hilbert_dim = host_layout.hilbert_dim;
    cuda_layout.d_target_desc = nullptr;
    cuda_layout.d_complement_desc = nullptr;

    // Full Hilbert stride per site (site-0-major): stride[i] = prod_{j>i} dims[j].
    std::vector<Index> full_stride(num_sites, 1);
    for (Index i = num_sites; i-- > 1;) {
        full_stride[i - 1] = full_stride[i] * dims[i];
    }

    std::vector<bool> is_target(num_sites, false);
    for (Index s : target_sites) {
        is_target[s] = true;
    }
    std::vector<Index> complement_sites;
    for (Index s = 0; s < num_sites; ++s) {
        if (!is_target[s]) {
            complement_sites.push_back(s);
        }
    }

    // Packed descriptor [full_stride, dim, weight] per position; weight[p] = prod of
    // subsequent dims in the subset (matches flatten_subset ordering in the host layout).
    auto build_desc = [&](const std::vector<Index>& subset) {
        const Index n = subset.size();
        std::vector<Index> desc(3 * n, 0);
        Index weight = 1;
        for (Index p = n; p-- > 0;) {
            desc[3 * p + 0] = full_stride[subset[p]];
            desc[3 * p + 1] = dims[subset[p]];
            desc[3 * p + 2] = weight;
            weight *= dims[subset[p]];
        }
        return desc;
    };
    const std::vector<Index> tdesc = build_desc(target_sites);
    const std::vector<Index> cdesc = build_desc(complement_sites);

    Index target_dim = 1;
    for (Index s : target_sites) {
        target_dim *= dims[s];
    }
    Index complement_dim = 1;
    for (Index s : complement_sites) {
        complement_dim *= dims[s];
    }
    cuda_layout.target_dim = target_dim;
    cuda_layout.complement_dim = complement_dim;
    cuda_layout.num_target = target_sites.size();
    cuda_layout.num_complement = complement_sites.size();

    auto upload = [](Index** dptr, const std::vector<Index>& host) -> bool {
        if (host.empty()) {
            *dptr = nullptr;
            return true;
        }
        if (cudaMalloc(reinterpret_cast<void**>(dptr), host.size() * sizeof(Index)) != cudaSuccess) {
            *dptr = nullptr;
            return false;
        }
        if (cudaMemcpy(*dptr, host.data(), host.size() * sizeof(Index),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(*dptr);
            *dptr = nullptr;
            return false;
        }
        return true;
    };

    if (!upload(&cuda_layout.d_target_desc, tdesc)) {
        return false;
    }
    if (!upload(&cuda_layout.d_complement_desc, cdesc)) {
        if (cuda_layout.d_target_desc != nullptr) {
            cudaFree(cuda_layout.d_target_desc);
            cuda_layout.d_target_desc = nullptr;
        }
        return false;
    }
    return true;
}

bool destroy_cuda_grouped_state_layout(
    CudaGroupedStateLayout& cuda_layout)
{
    bool ok = true;

    if (cuda_layout.d_complement_desc != nullptr &&
        cudaFree(cuda_layout.d_complement_desc) != cudaSuccess) {
        ok = false;
    }
    if (cuda_layout.d_target_desc != nullptr &&
        cudaFree(cuda_layout.d_target_desc) != cudaSuccess) {
        ok = false;
    }

    cuda_layout.d_complement_desc = nullptr;
    cuda_layout.d_target_desc = nullptr;
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
        cuda_layout.flat_size,
        cuda_layout.hilbert_dim, cuda_layout.target_dim, cuda_layout.complement_dim,
        cuda_layout.d_target_desc, cuda_layout.num_target,
        cuda_layout.d_complement_desc, cuda_layout.num_complement);

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
        cuda_layout.grouped_size,
        cuda_layout.hilbert_dim, cuda_layout.target_dim, cuda_layout.complement_dim,
        cuda_layout.d_target_desc, cuda_layout.num_target,
        cuda_layout.d_complement_desc, cuda_layout.num_complement);

    return cudaGetLastError() == cudaSuccess;
}

bool launch_flat_to_grouped_batched_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    Index batch_size,
    const void* d_flat_input,
    void* d_grouped_output,
    cudaStream_t stream)
{
    constexpr int block_size = 256;
    const Index total_flat_size = batch_size * cuda_layout.flat_size;
    const int grid_size =
        static_cast<int>((total_flat_size + block_size - 1) / block_size);

    flat_to_grouped_batched_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const Complex*>(d_flat_input),
        reinterpret_cast<Complex*>(d_grouped_output),
        cuda_layout.flat_size,
        cuda_layout.grouped_size,
        total_flat_size,
        cuda_layout.hilbert_dim, cuda_layout.target_dim, cuda_layout.complement_dim,
        cuda_layout.d_target_desc, cuda_layout.num_target,
        cuda_layout.d_complement_desc, cuda_layout.num_complement);

    return cudaGetLastError() == cudaSuccess;
}

bool launch_grouped_to_flat_batched_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    Index batch_size,
    const void* d_grouped_input,
    void* d_flat_output,
    cudaStream_t stream)
{
    constexpr int block_size = 256;
    const Index total_grouped_size = batch_size * cuda_layout.grouped_size;
    const int grid_size =
        static_cast<int>((total_grouped_size + block_size - 1) / block_size);

    grouped_to_flat_batched_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const Complex*>(d_grouped_input),
        reinterpret_cast<Complex*>(d_flat_output),
        cuda_layout.flat_size,
        cuda_layout.grouped_size,
        total_grouped_size,
        cuda_layout.hilbert_dim, cuda_layout.target_dim, cuda_layout.complement_dim,
        cuda_layout.d_target_desc, cuda_layout.num_target,
        cuda_layout.d_complement_desc, cuda_layout.num_complement);

    return cudaGetLastError() == cudaSuccess;
}

} // namespace culindblad

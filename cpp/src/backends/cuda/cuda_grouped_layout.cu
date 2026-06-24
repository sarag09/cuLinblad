#include "culindblad/cuda_grouped_layout.hpp"

#include <cstddef>
#include <cuComplex.h>
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

// ---- State-vector (ket-only) variants. ----
// For a length-D state vector |psi> the layout carries only the KET index: the flat index IS
// ket_full (no /D,%D bra split), and the grouped index collapses to g = ket_t*complement_dim +
// ket_c (the bra_t/bra_c axes are gone). The target/complement descriptors are identical to the
// density case (they describe the ket decomposition, which is bra-agnostic).
__device__ inline Index flat_to_grouped_vec_index(
    Index ket_full, Index complement_dim,
    const Index* tdesc, Index num_t, const Index* cdesc, Index num_c)
{
    Index ket_t = 0, ket_c = 0;
    for (Index p = 0; p < num_t; ++p) {
        const Index fs = tdesc[3 * p], dm = tdesc[3 * p + 1], w = tdesc[3 * p + 2];
        ket_t += ((ket_full / fs) % dm) * w;
    }
    for (Index q = 0; q < num_c; ++q) {
        const Index fs = cdesc[3 * q], dm = cdesc[3 * q + 1], w = cdesc[3 * q + 2];
        ket_c += ((ket_full / fs) % dm) * w;
    }
    return ket_t * complement_dim + ket_c;
}

__device__ inline Index grouped_to_flat_vec_index(
    Index g, Index complement_dim,
    const Index* tdesc, Index num_t, const Index* cdesc, Index num_c)
{
    const Index ket_c = g % complement_dim;
    const Index ket_t = g / complement_dim;
    Index ket_full = 0;
    for (Index p = 0; p < num_t; ++p) {
        const Index fs = tdesc[3 * p], dm = tdesc[3 * p + 1], w = tdesc[3 * p + 2];
        ket_full += ((ket_t / w) % dm) * fs;
    }
    for (Index q = 0; q < num_c; ++q) {
        const Index fs = cdesc[3 * q], dm = cdesc[3 * q + 1], w = cdesc[3 * q + 2];
        ket_full += ((ket_c / w) % dm) * fs;
    }
    return ket_full;
}

__global__ void flat_to_grouped_vec_batched_kernel(
    const Complex* flat_input,
    Complex* grouped_output,
    Index flat_size,
    Index grouped_size,
    Index total_flat_size,
    Index complement_dim,
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
    const Index g = flat_to_grouped_vec_index(local_flat, complement_dim,
                                              tdesc, num_t, cdesc, num_c);
    grouped_output[batch * grouped_size + g] = flat_input[idx];
}

__global__ void grouped_to_flat_vec_batched_kernel(
    const Complex* grouped_input,
    Complex* flat_output,
    Index flat_size,
    Index grouped_size,
    Index total_grouped_size,
    Index complement_dim,
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
    const Index f = grouped_to_flat_vec_index(local_grouped, complement_dim,
                                              tdesc, num_t, cdesc, num_c);
    flat_output[batch * flat_size + f] = grouped_input[idx];
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

// Like grouped_to_flat_batched_kernel but ACCUMULATES into the flat output
// (flat[f] += grouped[g]) so a layer's contribution can be added straight into the RHS
// vector f (zeroed once per RHS step), removing the per-layer temp vector + add + copy.
// Each grouped index maps to a unique flat index within a call, so there is no intra-call
// write conflict; sequential layers accumulate in stream order.
__global__ void grouped_to_flat_accumulate_batched_kernel(
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
    const Index out = batch * flat_size + f;
    cuDoubleComplex* fo = reinterpret_cast<cuDoubleComplex*>(flat_output);
    const cuDoubleComplex* gi = reinterpret_cast<const cuDoubleComplex*>(grouped_input);
    fo[out] = cuCadd(fo[out], gi[idx]);
}

// Fused: compute -i[H_sparse, rho] for a 2-site sparse coupling and ACCUMULATE it straight
// into the flat RHS vector f (f[g->flat] += val), skipping the grouped accumulator write +
// the separate grouped_to_flat flatten pass. Used only when the sparse commutator is the
// SOLE contributor on the layer (pure couplings, no dissipator). The commutator value is
// computed exactly as batched_grouped_sparse_commutator_kernel (cuda_elementwise.cu) and the
// grouped->flat index map is a bijection, so the result equals the separate-path result
// (only the float-accumulation order across layers may differ -> rounding-level).
__global__ void batched_grouped_sparse_commutator_to_flat_kernel(
    const cuDoubleComplex* sparse_values,
    const Index* sparse_rows,
    const Index* sparse_cols,
    Index sparse_nnz,
    const cuDoubleComplex* grouped_input,
    cuDoubleComplex* flat_output,
    Index flat_size,
    Index grouped_size,
    Index target_hilbert_dim,
    Index complement_dim,
    Index batch_size,
    Index D,
    const Index* tdesc, Index num_t, const Index* cdesc, Index num_c)
{
    const Index total = batch_size * grouped_size;
    const Index idx =
        static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x) +
        static_cast<Index>(threadIdx.x);
    if (idx >= total) {
        return;
    }
    const Index batch = idx / grouped_size;
    const Index local = idx % grouped_size;

    const Index bra_comp = local % complement_dim;
    Index t = local / complement_dim;
    const Index bra_target_out = t % target_hilbert_dim;
    t /= target_hilbert_dim;
    const Index ket_comp = t % complement_dim;
    const Index ket_target_out = t / complement_dim;

    const Index batch_offset = batch * grouped_size;
    const Index left_base =
        batch_offset +
        (ket_comp * target_hilbert_dim + bra_target_out) * complement_dim + bra_comp;
    const Index right_base =
        batch_offset +
        (ket_target_out * complement_dim + ket_comp) * target_hilbert_dim * complement_dim +
        bra_comp;
    const Index left_stride = complement_dim * target_hilbert_dim * complement_dim;
    const Index right_stride = complement_dim;

    cuDoubleComplex left_sum = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex right_sum = make_cuDoubleComplex(0.0, 0.0);
    for (Index nz = 0; nz < sparse_nnz; ++nz) {
        const Index row = sparse_rows[nz];
        const Index col = sparse_cols[nz];
        const cuDoubleComplex value = sparse_values[nz];
        if (row == ket_target_out) {
            left_sum = cuCadd(left_sum,
                cuCmul(value, grouped_input[left_base + col * left_stride]));
        }
        if (col == bra_target_out) {
            right_sum = cuCadd(right_sum,
                cuCmul(grouped_input[right_base + row * right_stride], value));
        }
    }
    const cuDoubleComplex minus_i = make_cuDoubleComplex(0.0, -1.0);
    const cuDoubleComplex val = cuCmul(minus_i, cuCsub(left_sum, right_sum));

    const Index f = grouped_to_flat_index(local, D, target_hilbert_dim, complement_dim,
                                          tdesc, num_t, cdesc, num_c);
    const Index out = batch * flat_size + f;
    flat_output[out] = cuCadd(flat_output[out], val);
}

// Fused single-site diagonal (commutator + anti-commutator + one diagonal jump) accumulated
// straight into f. Mirrors batched_grouped_fused_diagonal_kernel (cuda_elementwise.cu) but
// scatters into the flat RHS instead of the grouped accumulator.
__global__ void batched_grouped_fused_diagonal_to_flat_kernel(
    const cuDoubleComplex* h_diag,
    const cuDoubleComplex* norm_diag,
    const cuDoubleComplex* jump_diag,
    const cuDoubleComplex* grouped_input,
    cuDoubleComplex* flat_output,
    Index flat_size,
    Index grouped_size,
    Index target_hilbert_dim,
    Index complement_dim,
    Index batch_size,
    Index D,
    const Index* tdesc, Index num_t, const Index* cdesc, Index num_c)
{
    const Index total = batch_size * grouped_size;
    const Index idx =
        static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x) +
        static_cast<Index>(threadIdx.x);
    if (idx >= total) {
        return;
    }
    const Index batch = idx / grouped_size;
    const Index local = idx % grouped_size;
    const Index bra_comp = local % complement_dim;
    Index t = local / complement_dim;
    const Index bra_target = t % target_hilbert_dim;
    t /= target_hilbert_dim;
    const Index ket_comp = t % complement_dim;
    const Index ket_target = t / complement_dim;
    (void)bra_comp;
    (void)ket_comp;

    cuDoubleComplex factor = make_cuDoubleComplex(0.0, 0.0);
    if (h_diag != nullptr) {
        const cuDoubleComplex minus_i = make_cuDoubleComplex(0.0, -1.0);
        factor = cuCadd(factor,
            cuCmul(minus_i, cuCsub(h_diag[ket_target], h_diag[bra_target])));
    }
    if (norm_diag != nullptr) {
        const cuDoubleComplex minus_half = make_cuDoubleComplex(-0.5, 0.0);
        factor = cuCadd(factor,
            cuCmul(minus_half, cuCadd(norm_diag[ket_target], norm_diag[bra_target])));
    }
    if (jump_diag != nullptr) {
        factor = cuCadd(factor,
            cuCmul(jump_diag[ket_target], cuConj(jump_diag[bra_target])));
    }
    const cuDoubleComplex val = cuCmul(factor, grouped_input[idx]);
    const Index f = grouped_to_flat_index(local, D, target_hilbert_dim, complement_dim,
                                          tdesc, num_t, cdesc, num_c);
    const Index out = batch * flat_size + f;
    flat_output[out] = cuCadd(flat_output[out], val);
}

// Single-site dense jump L rho L^dag accumulated straight into f. Mirrors
// batched_grouped_dense_jump_kernel (cuda_elementwise.cu) but scatters into the flat RHS.
__global__ void batched_grouped_dense_jump_to_flat_kernel(
    const cuDoubleComplex* l_op,
    const cuDoubleComplex* grouped_input,
    cuDoubleComplex* flat_output,
    Index flat_size,
    Index grouped_size,
    Index target_hilbert_dim,
    Index complement_dim,
    Index batch_size,
    Index D,
    const Index* tdesc, Index num_t, const Index* cdesc, Index num_c)
{
    const Index total = batch_size * grouped_size;
    const Index idx =
        static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x) +
        static_cast<Index>(threadIdx.x);
    if (idx >= total) {
        return;
    }
    const Index batch = idx / grouped_size;
    const Index local = idx % grouped_size;
    const Index bra_comp = local % complement_dim;
    Index t = local / complement_dim;
    const Index bra_target = t % target_hilbert_dim;
    t /= target_hilbert_dim;
    const Index ket_comp = t % complement_dim;
    const Index ket_target = t / complement_dim;

    const Index base = batch * grouped_size;
    cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
    for (Index k = 0; k < target_hilbert_dim; ++k) {
        const cuDoubleComplex l_ik = l_op[ket_target * target_hilbert_dim + k];
        if (l_ik.x == 0.0 && l_ik.y == 0.0) {
            continue;
        }
        for (Index l = 0; l < target_hilbert_dim; ++l) {
            const cuDoubleComplex l_jl = l_op[bra_target * target_hilbert_dim + l];
            if (l_jl.x == 0.0 && l_jl.y == 0.0) {
                continue;
            }
            const Index rho_idx =
                base +
                ((k * complement_dim + ket_comp) * target_hilbert_dim + l) * complement_dim +
                bra_comp;
            acc = cuCadd(acc,
                cuCmul(cuCmul(l_ik, grouped_input[rho_idx]), cuConj(l_jl)));
        }
    }
    const Index f = grouped_to_flat_index(local, D, target_hilbert_dim, complement_dim,
                                          tdesc, num_t, cdesc, num_c);
    const Index out = batch * flat_size + f;
    flat_output[out] = cuCadd(flat_output[out], acc);
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

// State-vector layout: identical descriptors to the density layout (the target/complement
// decomposition is bra-agnostic), but the buffer holds D elements (a ket vector), not D^2.
// Overriding only flat_size/grouped_size keeps the D-vs-D^2 distinction localized and lets the
// vec relayout/contraction launchers read cuda_layout.flat_size like the density ones do.
bool create_cuda_psi_grouped_state_layout(
    const GroupedStateLayout& host_layout,
    CudaGroupedStateLayout& cuda_layout)
{
    if (!create_cuda_grouped_state_layout(host_layout, cuda_layout)) {
        return false;
    }
    cuda_layout.flat_size = host_layout.hilbert_dim;
    cuda_layout.grouped_size = host_layout.hilbert_dim;
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

bool launch_flat_to_grouped_vec_batched_kernel(
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

    flat_to_grouped_vec_batched_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const Complex*>(d_flat_input),
        reinterpret_cast<Complex*>(d_grouped_output),
        cuda_layout.flat_size,
        cuda_layout.grouped_size,
        total_flat_size,
        cuda_layout.complement_dim,
        cuda_layout.d_target_desc, cuda_layout.num_target,
        cuda_layout.d_complement_desc, cuda_layout.num_complement);

    return cudaGetLastError() == cudaSuccess;
}

bool launch_grouped_to_flat_vec_batched_kernel(
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

    grouped_to_flat_vec_batched_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const Complex*>(d_grouped_input),
        reinterpret_cast<Complex*>(d_flat_output),
        cuda_layout.flat_size,
        cuda_layout.grouped_size,
        total_grouped_size,
        cuda_layout.complement_dim,
        cuda_layout.d_target_desc, cuda_layout.num_target,
        cuda_layout.d_complement_desc, cuda_layout.num_complement);

    return cudaGetLastError() == cudaSuccess;
}

bool launch_grouped_to_flat_accumulate_batched_kernel(
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

    grouped_to_flat_accumulate_batched_kernel<<<grid_size, block_size, 0, stream>>>(
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

bool launch_grouped_sparse_commutator_to_flat_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    const void* sparse_values,
    const void* sparse_rows,
    const void* sparse_cols,
    Index sparse_nnz,
    Index batch_size,
    const void* d_grouped_input,
    void* d_flat_output,
    cudaStream_t stream)
{
    constexpr int block_size = 256;
    const Index total = batch_size * cuda_layout.grouped_size;
    const int grid_size = static_cast<int>((total + block_size - 1) / block_size);

    batched_grouped_sparse_commutator_to_flat_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const cuDoubleComplex*>(sparse_values),
        reinterpret_cast<const Index*>(sparse_rows),
        reinterpret_cast<const Index*>(sparse_cols),
        sparse_nnz,
        reinterpret_cast<const cuDoubleComplex*>(d_grouped_input),
        reinterpret_cast<cuDoubleComplex*>(d_flat_output),
        cuda_layout.flat_size,
        cuda_layout.grouped_size,
        cuda_layout.target_dim,
        cuda_layout.complement_dim,
        batch_size,
        cuda_layout.hilbert_dim,
        cuda_layout.d_target_desc, cuda_layout.num_target,
        cuda_layout.d_complement_desc, cuda_layout.num_complement);

    return cudaGetLastError() == cudaSuccess;
}

bool launch_grouped_fused_diagonal_to_flat_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    const void* h_diag,
    const void* norm_diag,
    const void* jump_diag,
    Index batch_size,
    const void* d_grouped_input,
    void* d_flat_output,
    cudaStream_t stream)
{
    constexpr int block_size = 256;
    const Index total = batch_size * cuda_layout.grouped_size;
    const int grid_size = static_cast<int>((total + block_size - 1) / block_size);

    batched_grouped_fused_diagonal_to_flat_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const cuDoubleComplex*>(h_diag),
        reinterpret_cast<const cuDoubleComplex*>(norm_diag),
        reinterpret_cast<const cuDoubleComplex*>(jump_diag),
        reinterpret_cast<const cuDoubleComplex*>(d_grouped_input),
        reinterpret_cast<cuDoubleComplex*>(d_flat_output),
        cuda_layout.flat_size,
        cuda_layout.grouped_size,
        cuda_layout.target_dim,
        cuda_layout.complement_dim,
        batch_size,
        cuda_layout.hilbert_dim,
        cuda_layout.d_target_desc, cuda_layout.num_target,
        cuda_layout.d_complement_desc, cuda_layout.num_complement);

    return cudaGetLastError() == cudaSuccess;
}

bool launch_grouped_dense_jump_to_flat_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    const void* l_op,
    Index batch_size,
    const void* d_grouped_input,
    void* d_flat_output,
    cudaStream_t stream)
{
    constexpr int block_size = 256;
    const Index total = batch_size * cuda_layout.grouped_size;
    const int grid_size = static_cast<int>((total + block_size - 1) / block_size);

    batched_grouped_dense_jump_to_flat_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const cuDoubleComplex*>(l_op),
        reinterpret_cast<const cuDoubleComplex*>(d_grouped_input),
        reinterpret_cast<cuDoubleComplex*>(d_flat_output),
        cuda_layout.flat_size,
        cuda_layout.grouped_size,
        cuda_layout.target_dim,
        cuda_layout.complement_dim,
        batch_size,
        cuda_layout.hilbert_dim,
        cuda_layout.d_target_desc, cuda_layout.num_target,
        cuda_layout.d_complement_desc, cuda_layout.num_complement);

    return cudaGetLastError() == cudaSuccess;
}

} // namespace culindblad

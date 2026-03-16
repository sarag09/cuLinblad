#pragma once

#include <vector>
#include <utility>

#include "culindblad/types.hpp"
#include "culindblad/local_dims.hpp"

namespace culindblad {

struct StateLayout {
    std::vector<Index> local_dims;
    std::vector<Index> ket_strides;
    std::vector<Index> bra_strides;
    Index hilbert_dim;
    Index density_dim;
};

inline std::vector<Index> compute_strides(const std::vector<Index>& dims)
{
    std::vector<Index> strides(dims.size(), 1);

    if (dims.empty()) {
        return strides;
    }

    for (Index i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }

    return strides;
}

inline StateLayout make_state_layout(const std::vector<Index>& local_dims)
{
    StateLayout layout;
    layout.local_dims = local_dims;
    layout.ket_strides = compute_strides(local_dims);
    layout.bra_strides = compute_strides(local_dims);
    layout.hilbert_dim = total_hilbert_dim(local_dims);
    layout.density_dim = layout.hilbert_dim * layout.hilbert_dim;
    return layout;
}

inline Index flatten_ket_index(const std::vector<Index>& subsys_index, const std::vector<Index>& strides)
{
    Index result = 0;

    for (Index i = 0; i < subsys_index.size(); ++i) {
        result += subsys_index[i] * strides[i];
    }

    return result;
}

inline std::vector<Index> unflatten_ket_index(
    Index flat_index,
    const std::vector<Index>& strides,
    const std::vector<Index>& dims)
{
    std::vector<Index> result(strides.size(), 0);

    for (Index i = 0; i < strides.size(); ++i) {
        result[i] = flat_index / strides[i];
        flat_index = flat_index % strides[i];
    }

    return result;
}

inline Index flatten_density_index(
    Index ket_index,
    Index bra_index,
    Index hilbert_dim)
{
    return ket_index * hilbert_dim + bra_index;
}

inline std::pair<Index, Index> unflatten_density_index(
    Index flat_index,
    Index hilbert_dim)
{
    Index ket_index = flat_index / hilbert_dim;
    Index bra_index = flat_index % hilbert_dim;
    return {ket_index, bra_index};
}

} // namespace culindblad

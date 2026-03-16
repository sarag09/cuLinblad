#pragma once

#include <vector>

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

} // namespace culindblad

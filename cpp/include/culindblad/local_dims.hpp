#pragma once

#include <numeric>
#include <vector>

#include "culindblad/types.hpp"

namespace culindblad {

inline Index num_subsystems(const std::vector<Index>& local_dims)
{
    return local_dims.size();
}

inline Index total_hilbert_dim(const std::vector<Index>& local_dims)
{
    Index result = 1;

    for (Index dim : local_dims) {
        result *= dim;
    }

    return result;
}

} // namespace culindblad

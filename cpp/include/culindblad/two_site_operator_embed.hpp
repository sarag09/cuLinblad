#pragma once

#include <vector>

#include "culindblad/types.hpp"

namespace culindblad {

std::vector<Complex> embed_two_site_operator(
    const std::vector<Complex>& local_op,
    Index dim_site_a,
    Index dim_site_b,
    Index site_a,
    Index site_b,
    const std::vector<Index>& local_dims);

} // namespace culindblad

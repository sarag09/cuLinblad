#pragma once

#include <vector>

#include "culindblad/types.hpp"

namespace culindblad {

std::vector<Complex> embed_k_site_operator(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& sites,
    const std::vector<Index>& local_dims);

} // namespace culindblad

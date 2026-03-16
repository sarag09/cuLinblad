#pragma once

#include <vector>

#include "culindblad/types.hpp"

namespace culindblad {

std::vector<Complex> multiply_square_matrices(
    const std::vector<Complex>& A,
    const std::vector<Complex>& B,
    Index dim);

} // namespace culindblad

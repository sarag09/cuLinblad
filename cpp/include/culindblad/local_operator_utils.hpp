#pragma once

#include <vector>

#include "culindblad/types.hpp"

namespace culindblad {

std::vector<Complex> local_conjugate_transpose(
    const std::vector<Complex>& op,
    Index dim);

std::vector<Complex> local_multiply_square(
    const std::vector<Complex>& A,
    const std::vector<Complex>& B,
    Index dim);

} // namespace culindblad

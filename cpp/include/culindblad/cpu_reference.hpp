#pragma once

#include <vector>

#include "culindblad/state_buffer.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

std::vector<Complex> multiply_square_matrices(
    ConstStateBuffer A,
    ConstStateBuffer B,
    Index dim);

} // namespace culindblad
#pragma once

#include <vector>

#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

inline void apply_liouvillian(
    const Solver& solver,
    const std::vector<Complex>& rho_in,
    std::vector<Complex>& rho_out)
{
    rho_out = rho_in;
}

} // namespace culindblad

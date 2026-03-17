#pragma once

#include <vector>

#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

void apply_liouvillian_cpu_reference(
    const Solver& solver,
    const std::vector<Complex>& rho_in,
    std::vector<Complex>& rho_out);

} // namespace culindblad

#pragma once

#include "culindblad/solver.hpp"
#include "culindblad/state_buffer.hpp"

namespace culindblad {

void apply_liouvillian(
    const Solver& solver,
    ConstStateBuffer rho_in,
    StateBuffer rho_out);

} // namespace culindblad
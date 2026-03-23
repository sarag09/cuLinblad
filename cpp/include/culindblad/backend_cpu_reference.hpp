#pragma once

#include "culindblad/solver.hpp"
#include "culindblad/state_buffer.hpp"

namespace culindblad {

void apply_liouvillian_cpu_reference(
    const Solver& solver,
    ConstStateBuffer rho_in,
    StateBuffer rho_out);

void apply_liouvillian_cpu_reference_at_time(
    const Solver& solver,
    double t,
    ConstStateBuffer rho_in,
    StateBuffer rho_out);

} // namespace culindblad
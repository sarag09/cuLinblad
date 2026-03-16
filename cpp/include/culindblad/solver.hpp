#pragma once

#include "culindblad/model.hpp"
#include "culindblad/state_layout.hpp"

namespace culindblad {

struct Solver {
    Model model;
    StateLayout layout;
};

inline Solver make_solver(const Model& model)
{
    Solver solver;
    solver.model = model;
    solver.layout = make_state_layout(model.local_dims);
    return solver;
}

} // namespace culindblad

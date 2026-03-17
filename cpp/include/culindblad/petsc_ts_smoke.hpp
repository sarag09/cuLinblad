#pragma once

#include <petscts.h>

#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

PetscErrorCode run_ts_smoke_test(
    const Solver& solver,
    Index ket_index,
    Index bra_index,
    Complex& out_value);

} // namespace culindblad

#pragma once

#include <petscmat.h>

#include "culindblad/solver.hpp"

namespace culindblad {

PetscErrorCode matshell_apply(Mat A, Vec x, Vec y);

} // namespace culindblad

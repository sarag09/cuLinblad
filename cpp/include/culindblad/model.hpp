#pragma once

#include <vector>

#include "culindblad/types.hpp"
#include "culindblad/operator_term.hpp"

namespace culindblad {

struct Model {
    std::vector<Index> local_dims;
    std::vector<OperatorTerm> hamiltonian_terms;
    std::vector<OperatorTerm> dissipator_terms;
};

} // namespace culindblad

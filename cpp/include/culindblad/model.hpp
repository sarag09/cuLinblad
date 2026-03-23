#pragma once

#include <vector>

#include "culindblad/operator_term.hpp"
#include "culindblad/time_dependent_term.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

struct Model {
    std::vector<Index> local_dims;
    std::vector<OperatorTerm> hamiltonian_terms;
    std::vector<OperatorTerm> dissipator_terms;
    std::vector<TimeDependentTerm> time_dependent_hamiltonian_terms;
};

} // namespace culindblad
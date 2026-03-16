#pragma once

#include <string>
#include <vector>

#include "culindblad/types.hpp"

namespace culindblad {

enum class TermKind {
    Hamiltonian,
    Dissipator
};

struct OperatorTerm {
    TermKind kind;
    std::string name;
    std::vector<Index> sites;
    std::vector<Complex> matrix;
    Index row_dim;
    Index col_dim;
};

} // namespace culindblad

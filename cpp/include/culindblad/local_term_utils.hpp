#pragma once

#include <vector>

#include "culindblad/operator_term.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

bool term_is_full_dense(
    const OperatorTerm& term,
    Index hilbert_dim);

bool term_is_local_k_site(
    const OperatorTerm& term,
    const std::vector<Index>& local_dims);

Index term_local_dimension(
    const OperatorTerm& term,
    const std::vector<Index>& local_dims);

} // namespace culindblad

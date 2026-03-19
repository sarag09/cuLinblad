#include <stdexcept>
#include <vector>

#include "culindblad/local_term_utils.hpp"
#include "culindblad/operator_term.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

bool term_is_full_dense(
    const OperatorTerm& term,
    Index hilbert_dim)
{
    return (term.row_dim == hilbert_dim) && (term.col_dim == hilbert_dim);
}

Index term_local_dimension(
    const OperatorTerm& term,
    const std::vector<Index>& local_dims)
{
    if (term.sites.empty()) {
        throw std::runtime_error("term_local_dimension: term has no sites");
    }

    Index dim = 1;
    for (Index site : term.sites) {
        if (site >= local_dims.size()) {
            throw std::runtime_error("term_local_dimension: site out of range");
        }
        dim *= local_dims[site];
    }

    return dim;
}

bool term_is_local_k_site(
    const OperatorTerm& term,
    const std::vector<Index>& local_dims)
{
    if (term.sites.empty()) {
        return false;
    }

    const Index dim = term_local_dimension(term, local_dims);
    return (term.row_dim == dim) && (term.col_dim == dim);
}

} // namespace culindblad

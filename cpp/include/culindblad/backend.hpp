#pragma once

#include <stdexcept>
#include <vector>

#include "culindblad/liouvillian_terms.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

inline void apply_liouvillian(
    const Solver& solver,
    const std::vector<Complex>& rho_in,
    std::vector<Complex>& rho_out)
{
    if (rho_in.size() != solver.layout.density_dim) {
        throw std::runtime_error("apply_liouvillian: rho_in has wrong size");
    }

    if (!solver.model.dissipator_terms.empty()) {
        throw std::runtime_error("apply_liouvillian: dissipators are not implemented yet");
    }

    if (solver.model.hamiltonian_terms.size() != 1) {
        throw std::runtime_error(
            "apply_liouvillian: expected exactly one Hamiltonian term in this prototype");
    }

    const OperatorTerm& H_term = solver.model.hamiltonian_terms.at(0);

    if (H_term.row_dim != solver.layout.hilbert_dim ||
        H_term.col_dim != solver.layout.hilbert_dim) {
        throw std::runtime_error(
            "apply_liouvillian: Hamiltonian term must be a full-system dense operator");
    }

    rho_out = apply_hamiltonian_commutator(
        H_term.matrix,
        rho_in,
        solver.layout.hilbert_dim);
}

} // namespace culindblad
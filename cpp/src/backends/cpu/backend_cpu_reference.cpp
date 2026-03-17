#include <stdexcept>
#include <vector>

#include "culindblad/backend.hpp"
#include "culindblad/backend_cpu_reference.hpp"
#include "culindblad/liouvillian_terms.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

void apply_liouvillian_cpu_reference(
    const Solver& solver,
    const std::vector<Complex>& rho_in,
    std::vector<Complex>& rho_out)
{
    if (rho_in.size() != solver.layout.density_dim) {
        throw std::runtime_error("apply_liouvillian_cpu_reference: rho_in has wrong size");
    }

    rho_out.assign(solver.layout.density_dim, Complex{0.0, 0.0});

    for (const OperatorTerm& H_term : solver.model.hamiltonian_terms) {
        if (H_term.row_dim != solver.layout.hilbert_dim ||
            H_term.col_dim != solver.layout.hilbert_dim) {
            throw std::runtime_error(
                "apply_liouvillian_cpu_reference: Hamiltonian term must be a full-system dense operator");
        }

        const std::vector<Complex> contribution =
            apply_hamiltonian_commutator(
                H_term.matrix,
                rho_in,
                solver.layout.hilbert_dim);

        for (Index idx = 0; idx < solver.layout.density_dim; ++idx) {
            rho_out[idx] += contribution[idx];
        }
    }

    for (const OperatorTerm& L_term : solver.model.dissipator_terms) {
        if (L_term.row_dim != solver.layout.hilbert_dim ||
            L_term.col_dim != solver.layout.hilbert_dim) {
            throw std::runtime_error(
                "apply_liouvillian_cpu_reference: Dissipator term must be a full-system dense operator");
        }

        const std::vector<Complex> contribution =
            apply_dissipator(
                L_term.matrix,
                rho_in,
                solver.layout.hilbert_dim);

        for (Index idx = 0; idx < solver.layout.density_dim; ++idx) {
            rho_out[idx] += contribution[idx];
        }
    }
}

void apply_liouvillian(
    const Solver& solver,
    const std::vector<Complex>& rho_in,
    std::vector<Complex>& rho_out)
{
    apply_liouvillian_cpu_reference(solver, rho_in, rho_out);
}

} // namespace culindblad

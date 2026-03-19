#include <stdexcept>
#include <vector>

#include "culindblad/backend.hpp"
#include "culindblad/backend_cpu_reference.hpp"
#include "culindblad/liouvillian_terms.hpp"
#include "culindblad/local_apply.hpp"
#include "culindblad/local_term_utils.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

void apply_liouvillian_cpu_reference(
    const Solver& solver,
    ConstStateBuffer rho_in,
    StateBuffer rho_out)
{
    if (rho_in.size != solver.layout.density_dim) {
        throw std::runtime_error("apply_liouvillian_cpu_reference: rho_in has wrong size");
    }

    if (rho_out.size != solver.layout.density_dim) {
        throw std::runtime_error("apply_liouvillian_cpu_reference: rho_out has wrong size");
    }

    for (Index i = 0; i < rho_out.size; ++i) {
        rho_out[i] = Complex{0.0, 0.0};
    }

    for (const OperatorTerm& H_term : solver.model.hamiltonian_terms) {
        std::vector<Complex> contribution;

        if (term_is_local_k_site(H_term, solver.model.local_dims)) {
            contribution = apply_k_site_commutator(
                H_term.matrix,
                H_term.sites,
                solver.model.local_dims,
                rho_in);
        } else if (term_is_full_dense(H_term, solver.layout.hilbert_dim)) {
            contribution = apply_hamiltonian_commutator(
                H_term.matrix,
                rho_in,
                solver.layout.hilbert_dim);
        } else {
            throw std::runtime_error(
                "apply_liouvillian_cpu_reference: unsupported Hamiltonian term format");
        }

        for (Index idx = 0; idx < solver.layout.density_dim; ++idx) {
            rho_out[idx] += contribution[idx];
        }
    }

    for (const OperatorTerm& L_term : solver.model.dissipator_terms) {
        std::vector<Complex> contribution;

        if (term_is_local_k_site(L_term, solver.model.local_dims)) {
            contribution = apply_k_site_dissipator(
                L_term.matrix,
                L_term.sites,
                solver.model.local_dims,
                rho_in);
        } else if (term_is_full_dense(L_term, solver.layout.hilbert_dim)) {
            contribution = apply_dissipator(
                L_term.matrix,
                rho_in,
                solver.layout.hilbert_dim);
        } else {
            throw std::runtime_error(
                "apply_liouvillian_cpu_reference: unsupported dissipator term format");
        }

        for (Index idx = 0; idx < solver.layout.density_dim; ++idx) {
            rho_out[idx] += contribution[idx];
        }
    }
}

void apply_liouvillian(
    const Solver& solver,
    ConstStateBuffer rho_in,
    StateBuffer rho_out)
{
    apply_liouvillian_cpu_reference(solver, rho_in, rho_out);
}

} // namespace culindblad
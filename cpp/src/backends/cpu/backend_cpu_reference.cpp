#include <stdexcept>
#include <vector>

#include "culindblad/backend.hpp"
#include "culindblad/backend_cpu_reference.hpp"
#include "culindblad/liouvillian_terms.hpp"
#include "culindblad/local_apply.hpp"
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

        const bool is_one_site_local =
            (H_term.sites.size() == 1) &&
            (H_term.row_dim == solver.model.local_dims.at(H_term.sites.at(0))) &&
            (H_term.col_dim == solver.model.local_dims.at(H_term.sites.at(0)));

        const bool is_full_dense =
            (H_term.row_dim == solver.layout.hilbert_dim) &&
            (H_term.col_dim == solver.layout.hilbert_dim);

        if (is_one_site_local) {
            contribution = apply_one_site_commutator(
                H_term.matrix,
                H_term.row_dim,
                H_term.sites.at(0),
                solver.model.local_dims,
                rho_in);
        } else if (is_full_dense) {
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

        const bool is_one_site_local =
            (L_term.sites.size() == 1) &&
            (L_term.row_dim == solver.model.local_dims.at(L_term.sites.at(0))) &&
            (L_term.col_dim == solver.model.local_dims.at(L_term.sites.at(0)));

        const bool is_full_dense =
            (L_term.row_dim == solver.layout.hilbert_dim) &&
            (L_term.col_dim == solver.layout.hilbert_dim);

        if (is_one_site_local) {
            contribution = apply_one_site_dissipator(
                L_term.matrix,
                L_term.row_dim,
                L_term.sites.at(0),
                solver.model.local_dims,
                rho_in);
        } else if (is_full_dense) {
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
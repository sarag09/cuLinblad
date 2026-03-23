#include <stdexcept>
#include <vector>

#include "culindblad/backend.hpp"
#include "culindblad/backend_cpu_reference.hpp"
#include "culindblad/grouped_contraction_backend.hpp"
#include "culindblad/liouvillian_terms.hpp"
#include "culindblad/local_term_utils.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/time_dependent_term.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

namespace {

std::vector<Complex> apply_hamiltonian_term_contribution(
    const OperatorTerm& H_term,
    const Solver& solver,
    ConstStateBuffer rho_in)
{
    if (term_is_local_k_site(H_term, solver.model.local_dims)) {
        return apply_grouped_commutator(
            H_term.matrix,
            H_term.sites,
            solver.model.local_dims,
            rho_in);
    }

    if (term_is_full_dense(H_term, solver.layout.hilbert_dim)) {
        return apply_hamiltonian_commutator(
            H_term.matrix,
            rho_in,
            solver.layout.hilbert_dim);
    }

    throw std::runtime_error(
        "apply_hamiltonian_term_contribution: unsupported Hamiltonian term format");
}

std::vector<Complex> apply_dissipator_term_contribution(
    const OperatorTerm& L_term,
    const Solver& solver,
    ConstStateBuffer rho_in)
{
    if (term_is_local_k_site(L_term, solver.model.local_dims)) {
        return apply_grouped_dissipator(
            L_term.matrix,
            L_term.sites,
            solver.model.local_dims,
            rho_in);
    }

    if (term_is_full_dense(L_term, solver.layout.hilbert_dim)) {
        return apply_dissipator(
            L_term.matrix,
            rho_in,
            solver.layout.hilbert_dim);
    }

    throw std::runtime_error(
        "apply_dissipator_term_contribution: unsupported dissipator term format");
}

void accumulate_scaled_contribution(
    const std::vector<Complex>& contribution,
    Complex scale,
    StateBuffer rho_out)
{
    if (contribution.size() != rho_out.size) {
        throw std::runtime_error(
            "accumulate_scaled_contribution: contribution size mismatch");
    }

    for (Index idx = 0; idx < rho_out.size; ++idx) {
        rho_out[idx] += scale * contribution[idx];
    }
}

} // namespace

void apply_liouvillian_cpu_reference_at_time(
    const Solver& solver,
    double t,
    ConstStateBuffer rho_in,
    StateBuffer rho_out)
{
    if (rho_in.size != solver.layout.density_dim) {
        throw std::runtime_error("apply_liouvillian_cpu_reference_at_time: rho_in has wrong size");
    }

    if (rho_out.size != solver.layout.density_dim) {
        throw std::runtime_error("apply_liouvillian_cpu_reference_at_time: rho_out has wrong size");
    }

    for (Index i = 0; i < rho_out.size; ++i) {
        rho_out[i] = Complex{0.0, 0.0};
    }

    for (const OperatorTerm& H_term : solver.model.hamiltonian_terms) {
        const std::vector<Complex> contribution =
            apply_hamiltonian_term_contribution(H_term, solver, rho_in);

        accumulate_scaled_contribution(
            contribution,
            Complex{1.0, 0.0},
            rho_out);
    }

    for (const TimeDependentTerm& td_term : solver.model.time_dependent_hamiltonian_terms) {
        const double coeff =
            evaluate_time_dependent_coefficient(td_term, t);

        const OperatorTerm H_term{
            TermKind::Hamiltonian,
            td_term.name,
            td_term.sites,
            td_term.matrix,
            td_term.rows,
            td_term.cols
        };

        const std::vector<Complex> contribution =
            apply_hamiltonian_term_contribution(H_term, solver, rho_in);

        accumulate_scaled_contribution(
            contribution,
            Complex{coeff, 0.0},
            rho_out);
    }

    for (const OperatorTerm& L_term : solver.model.dissipator_terms) {
        const std::vector<Complex> contribution =
            apply_dissipator_term_contribution(L_term, solver, rho_in);

        accumulate_scaled_contribution(
            contribution,
            Complex{1.0, 0.0},
            rho_out);
    }
}

void apply_liouvillian_cpu_reference(
    const Solver& solver,
    ConstStateBuffer rho_in,
    StateBuffer rho_out)
{
    apply_liouvillian_cpu_reference_at_time(
        solver,
        0.0,
        rho_in,
        rho_out);
}

void apply_liouvillian(
    const Solver& solver,
    ConstStateBuffer rho_in,
    StateBuffer rho_out)
{
    apply_liouvillian_cpu_reference(solver, rho_in, rho_out);
}

void apply_liouvillian_at_time(
    const Solver& solver,
    double t,
    ConstStateBuffer rho_in,
    StateBuffer rho_out)
{
    apply_liouvillian_cpu_reference_at_time(
        solver,
        t,
        rho_in,
        rho_out);
}

} // namespace culindblad
#include <iostream>
#include <vector>

#include <petscts.h>

#include "culindblad/backend.hpp"
#include "culindblad/k_site_operator_embed.hpp"
#include "culindblad/local_apply.hpp"
#include "culindblad/model.hpp"
#include "culindblad/operator_term.hpp"
#include "culindblad/petsc_ts_smoke.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"
#include "culindblad/liouvillian_terms.hpp"

int main(int argc, char** argv)
{
    using namespace culindblad;

    PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
    if (ierr != 0) {
        std::cerr << "PetscInitialize failed." << std::endl;
        return 1;
    }

    const std::vector<Index> local_dims = {3, 3, 3, 3};

    std::vector<Complex> z_like = {
        Complex{1.0, 0.0},  Complex{0.0, 0.0},  Complex{0.0, 0.0},
        Complex{0.0, 0.0},  Complex{-1.0, 0.0}, Complex{0.0, 0.0},
        Complex{0.0, 0.0},  Complex{0.0, 0.0},  Complex{0.0, 0.0}
    };

    std::vector<Complex> zzz_three_site(27 * 27, Complex{0.0, 0.0});
    for (Index a = 0; a < 3; ++a) {
        for (Index b = 0; b < 3; ++b) {
            for (Index c = 0; c < 3; ++c) {
                const double za = (a == 0 ? 1.0 : (a == 1 ? -1.0 : 0.0));
                const double zb = (b == 0 ? 1.0 : (b == 1 ? -1.0 : 0.0));
                const double zc = (c == 0 ? 1.0 : (c == 1 ? -1.0 : 0.0));
                const Index idx = a * 9 + b * 3 + c;
                zzz_three_site[idx * 27 + idx] = Complex{za * zb * zc, 0.0};
            }
        }
    }

    std::vector<Complex> lowering_three_site(27 * 27, Complex{0.0, 0.0});
    lowering_three_site[0 * 27 + 9] = Complex{1.0, 0.0};

    OperatorTerm h_term{
        TermKind::Hamiltonian,
        "q1_q2_q3_zzz",
        {0, 1, 2},
        zzz_three_site,
        27,
        27
    };

    OperatorTerm d_term{
        TermKind::Dissipator,
        "q1_q2_q3_jump",
        {0, 1, 2},
        lowering_three_site,
        27,
        27
    };

    Model model{
        local_dims,
        {h_term},
        {d_term}
    };

    Solver solver = make_solver(model);

    std::cout << "cuLindblad k-site smoke test" << std::endl;
    std::cout << "Hilbert dimension: " << solver.layout.hilbert_dim << std::endl;
    std::cout << "Density dimension: " << solver.layout.density_dim << std::endl;

    std::vector<Complex> rho_in(solver.layout.density_dim, Complex{0.0, 0.0});
    std::vector<Complex> rho_out(solver.layout.density_dim, Complex{0.0, 0.0});

    rho_in[0 * solver.layout.hilbert_dim + 27] = Complex{1.0, 0.0};

    ConstStateBuffer in_buf{rho_in.data(), rho_in.size()};
    StateBuffer out_buf{rho_out.data(), rho_out.size()};

    apply_liouvillian(solver, in_buf, out_buf);

    std::cout << "Backend k-site total entry (0,27): "
              << rho_out.at(0 * solver.layout.hilbert_dim + 27) << std::endl;

    std::vector<Complex> local_comm =
        apply_k_site_commutator(zzz_three_site, {0, 1, 2}, local_dims, in_buf);

    std::vector<Complex> dense_h =
        embed_k_site_operator(zzz_three_site, {0, 1, 2}, local_dims);

    std::vector<Complex> dense_comm =
        apply_hamiltonian_commutator(dense_h, in_buf, solver.layout.hilbert_dim);

    std::cout << "k-site commutator entry (0,27): "
              << local_comm.at(0 * solver.layout.hilbert_dim + 27) << std::endl;
    std::cout << "Dense embedded commutator entry (0,27): "
              << dense_comm.at(0 * solver.layout.hilbert_dim + 27) << std::endl;
    std::cout << "Difference k-site-vs-dense commutator at (0,27): "
              << (local_comm.at(0 * solver.layout.hilbert_dim + 27)
                  - dense_comm.at(0 * solver.layout.hilbert_dim + 27))
              << std::endl;

    std::vector<Complex> local_diss =
        apply_k_site_dissipator(lowering_three_site, {0, 1, 2}, local_dims, in_buf);

    std::vector<Complex> dense_L =
        embed_k_site_operator(lowering_three_site, {0, 1, 2}, local_dims);

    std::vector<Complex> dense_diss =
        apply_dissipator(dense_L, in_buf, solver.layout.hilbert_dim);

    std::cout << "k-site dissipator entry (0,27): "
              << local_diss.at(0 * solver.layout.hilbert_dim + 27) << std::endl;
    std::cout << "Dense embedded dissipator entry (0,27): "
              << dense_diss.at(0 * solver.layout.hilbert_dim + 27) << std::endl;
    std::cout << "Difference k-site-vs-dense dissipator at (0,27): "
              << (local_diss.at(0 * solver.layout.hilbert_dim + 27)
                  - dense_diss.at(0 * solver.layout.hilbert_dim + 27))
              << std::endl;

    Complex ts_value{0.0, 0.0};
    ierr = run_ts_smoke_test(solver, 0, 27, ts_value);
    if (ierr != 0) {
        std::cerr << "run_ts_smoke_test failed." << std::endl;
        PetscFinalize();
        return 1;
    }

    std::cout << "TS evolved entry (0,27): " << ts_value << std::endl;

    PetscFinalize();
    return 0;
}
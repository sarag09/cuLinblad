#include <iostream>
#include <vector>

#include <petscts.h>

#include "culindblad/backend.hpp"
#include "culindblad/k_site_block_map.hpp"
#include "culindblad/k_site_grouped_apply.hpp"
#include "culindblad/k_site_operator_embed.hpp"
#include "culindblad/k_site_plan.hpp"
#include "culindblad/k_site_tensor_view.hpp"
#include "culindblad/liouvillian_terms.hpp"
#include "culindblad/model.hpp"
#include "culindblad/operator_term.hpp"
#include "culindblad/petsc_ts_smoke.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"

int main(int argc, char** argv)
{
    using namespace culindblad;

    PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
    if (ierr != 0) {
        std::cerr << "PetscInitialize failed." << std::endl;
        return 1;
    }

    const std::vector<Index> local_dims = {3, 3, 3, 3};

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

    std::cout << "cuLindblad grouped k-site smoke test" << std::endl;
    std::cout << "Hilbert dimension: " << solver.layout.hilbert_dim << std::endl;
    std::cout << "Density dimension: " << solver.layout.density_dim << std::endl;

    KSiteTensorView view = make_k_site_tensor_view({0, 1, 2}, local_dims);
    std::cout << "target dim product: " << view.ket_target_dim << std::endl;
    std::cout << "complement dim product: " << view.ket_complement_dim << std::endl;
    std::cout << "density target block size: " << view.density_target_block_size << std::endl;
    std::cout << "density complement block size: " << view.density_complement_block_size << std::endl;

    std::cout << "grouped sites: ";
    for (Index s : view.ket_grouped_sites) {
        std::cout << s << " ";
    }
    std::cout << std::endl;

    std::cout << "original-to-grouped positions: ";
    for (Index p : view.ket_original_to_grouped_position) {
        std::cout << p << " ";
    }
    std::cout << std::endl;

    KSiteTensorView view_nc = make_k_site_tensor_view({0, 2}, local_dims);
    std::cout << "\nnon-contiguous grouped sites: ";
    for (Index s : view_nc.ket_grouped_sites) {
        std::cout << s << " ";
    }
    std::cout << std::endl;

    std::cout << "non-contiguous original-to-grouped positions: ";
    for (Index p : view_nc.ket_original_to_grouped_position) {
        std::cout << p << " ";
    }
    std::cout << std::endl;

    KSiteBlockMap block_map = make_k_site_block_map({0, 1, 2}, local_dims);
    std::cout << "block map (target=0, comp=0) -> flat ket: "
              << block_map.grouped_to_flat_ket[0 * view.ket_complement_dim + 0] << std::endl;
    std::cout << "block map (target=9, comp=0) -> flat ket: "
              << block_map.grouped_to_flat_ket[9 * view.ket_complement_dim + 0] << std::endl;

    std::vector<Complex> rho_in(solver.layout.density_dim, Complex{0.0, 0.0});
    std::vector<Complex> rho_out(solver.layout.density_dim, Complex{0.0, 0.0});
    rho_in[0 * solver.layout.hilbert_dim + 27] = Complex{1.0, 0.0};

    ConstStateBuffer in_buf{rho_in.data(), rho_in.size()};
    StateBuffer out_buf{rho_out.data(), rho_out.size()};

    std::vector<Complex> grouped_left =
        apply_k_site_operator_left_grouped_reference(
            zzz_three_site, {0, 1, 2}, local_dims, in_buf);

    std::vector<Complex> grouped_right =
        apply_k_site_operator_right_grouped_reference(
            zzz_three_site, {0, 1, 2}, local_dims, in_buf);

    std::vector<Complex> grouped_comm =
        apply_k_site_commutator_grouped_reference(
            zzz_three_site, {0, 1, 2}, local_dims, in_buf);

    std::vector<Complex> grouped_diss =
        apply_k_site_dissipator_grouped_reference(
            lowering_three_site, {0, 1, 2}, local_dims, in_buf);

    std::vector<Complex> dense_h =
        embed_k_site_operator(zzz_three_site, {0, 1, 2}, local_dims);

    std::vector<Complex> dense_comm =
        apply_hamiltonian_commutator(dense_h, in_buf, solver.layout.hilbert_dim);

    std::vector<Complex> dense_L =
        embed_k_site_operator(lowering_three_site, {0, 1, 2}, local_dims);

    std::vector<Complex> dense_diss =
        apply_dissipator(dense_L, in_buf, solver.layout.hilbert_dim);

    std::cout << "Grouped left-action entry (0,27): "
              << grouped_left.at(0 * solver.layout.hilbert_dim + 27) << std::endl;
    std::cout << "Grouped right-action entry (0,27): "
              << grouped_right.at(0 * solver.layout.hilbert_dim + 27) << std::endl;

    std::cout << "Grouped commutator entry (0,27): "
              << grouped_comm.at(0 * solver.layout.hilbert_dim + 27) << std::endl;
    std::cout << "Dense embedded commutator entry (0,27): "
              << dense_comm.at(0 * solver.layout.hilbert_dim + 27) << std::endl;
    std::cout << "Difference grouped-vs-dense commutator at (0,27): "
              << (grouped_comm.at(0 * solver.layout.hilbert_dim + 27)
                  - dense_comm.at(0 * solver.layout.hilbert_dim + 27))
              << std::endl;

    std::cout << "Grouped dissipator entry (0,27): "
              << grouped_diss.at(0 * solver.layout.hilbert_dim + 27) << std::endl;
    std::cout << "Dense embedded dissipator entry (0,27): "
              << dense_diss.at(0 * solver.layout.hilbert_dim + 27) << std::endl;
    std::cout << "Difference grouped-vs-dense dissipator at (0,27): "
              << (grouped_diss.at(0 * solver.layout.hilbert_dim + 27)
                  - dense_diss.at(0 * solver.layout.hilbert_dim + 27))
              << std::endl;

    apply_liouvillian(solver, in_buf, out_buf);
    std::cout << "Backend k-site total entry (0,27): "
              << rho_out.at(0 * solver.layout.hilbert_dim + 27) << std::endl;

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
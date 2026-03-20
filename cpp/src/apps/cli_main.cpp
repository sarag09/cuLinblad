#include <iostream>
#include <vector>

#include <petscts.h>

#include "culindblad/backend.hpp"
#include "culindblad/grouped_contraction_backend.hpp"
#include "culindblad/k_site_block_map.hpp"
#include "culindblad/k_site_contraction_api.hpp"
#include "culindblad/k_site_contraction_desc.hpp"
#include "culindblad/k_site_index_roles.hpp"
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

    KSitePlan plan = make_k_site_plan({0, 1, 2}, local_dims);
    KSiteTensorView view = make_k_site_tensor_view({0, 1, 2}, local_dims);
    KSiteBlockMap block_map = make_k_site_block_map({0, 1, 2}, local_dims);
    KSiteContractionDesc desc = make_k_site_contraction_desc({0, 1, 2}, local_dims);
    KSiteIndexRoles roles = make_k_site_index_roles({0, 1, 2}, local_dims);
    GroupedContractionSpec left_spec =
        make_grouped_left_contraction_spec({0, 1, 2}, local_dims);
    GroupedContractionSpec right_spec =
        make_grouped_right_contraction_spec({0, 1, 2}, local_dims);

    std::cout << "target dim product: " << plan.target_dim_product << std::endl;
    std::cout << "complement dim product: " << plan.complement_dim_product << std::endl;

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
    std::cout << "non-contiguous grouped sites: ";
    for (Index s : view_nc.ket_grouped_sites) {
        std::cout << s << " ";
    }
    std::cout << std::endl;

    std::cout << "non-contiguous original-to-grouped positions: ";
    for (Index p : view_nc.ket_original_to_grouped_position) {
        std::cout << p << " ";
    }
    std::cout << std::endl;

    std::cout << "block map (target=0, comp=0) -> flat ket: "
              << block_map.grouped_to_flat_ket[0 * view.ket_complement_dim + 0] << std::endl;
    std::cout << "block map (target=9, comp=0) -> flat ket: "
              << block_map.grouped_to_flat_ket[9 * view.ket_complement_dim + 0] << std::endl;

    std::cout << "local contraction dim: " << desc.local_dim << std::endl;
    std::cout << "ket preserved dim: " << desc.ket_preserved_dim << std::endl;
    std::cout << "bra preserved dim: " << desc.bra_preserved_dim << std::endl;

    std::cout << "left input dims: ";
    for (Index d : desc.left_input_dims) {
        std::cout << d << " ";
    }
    std::cout << std::endl;

    std::cout << "right input dims: ";
    for (Index d : desc.right_input_dims) {
        std::cout << d << " ";
    }
    std::cout << std::endl;

    std::cout << "contracted target dim: " << roles.contracted_target_dim << std::endl;
    std::cout << "left target output dim: " << roles.left_target_output_dim << std::endl;
    std::cout << "left complement preserved dim: " << roles.left_complement_preserved_dim << std::endl;
    std::cout << "left bra total dim: " << roles.left_bra_total_dim << std::endl;
    std::cout << "right target output dim: " << roles.right_target_output_dim << std::endl;
    std::cout << "right complement preserved dim: " << roles.right_complement_preserved_dim << std::endl;
    std::cout << "right ket total dim: " << roles.right_ket_total_dim << std::endl;

    std::cout << "left contraction name: " << left_spec.contraction_name << std::endl;
    std::cout << "left contracted dim: " << left_spec.contracted_dim << std::endl;
    std::cout << "left preserved dim: " << left_spec.preserved_dim << std::endl;
    std::cout << "left passive full dim: " << left_spec.passive_full_dim << std::endl;

    std::cout << "right contraction name: " << right_spec.contraction_name << std::endl;
    std::cout << "right contracted dim: " << right_spec.contracted_dim << std::endl;
    std::cout << "right preserved dim: " << right_spec.preserved_dim << std::endl;
    std::cout << "right passive full dim: " << right_spec.passive_full_dim << std::endl;

    std::vector<Complex> rho_in(solver.layout.density_dim, Complex{0.0, 0.0});
    std::vector<Complex> rho_out(solver.layout.density_dim, Complex{0.0, 0.0});
    rho_in[0 * solver.layout.hilbert_dim + 27] = Complex{1.0, 0.0};

    ConstStateBuffer in_buf{rho_in.data(), rho_in.size()};
    StateBuffer out_buf{rho_out.data(), rho_out.size()};

    std::vector<Complex> grouped_left =
        apply_grouped_left_contraction(
            zzz_three_site, {0, 1, 2}, local_dims, in_buf);

    std::vector<Complex> grouped_right =
        apply_grouped_right_contraction(
            zzz_three_site, {0, 1, 2}, local_dims, in_buf);

    std::vector<Complex> grouped_comm =
        apply_grouped_commutator(
            zzz_three_site, {0, 1, 2}, local_dims, in_buf);

    std::vector<Complex> grouped_diss =
        apply_grouped_dissipator(
            lowering_three_site, {0, 1, 2}, local_dims, in_buf);

    std::vector<Complex> dense_h =
        embed_k_site_operator(zzz_three_site, {0, 1, 2}, local_dims);

    std::vector<Complex> dense_comm =
        apply_hamiltonian_commutator(dense_h, in_buf, solver.layout.hilbert_dim);

    std::vector<Complex> dense_L =
        embed_k_site_operator(lowering_three_site, {0, 1, 2}, local_dims);

    std::vector<Complex> dense_diss =
        apply_dissipator(dense_L, in_buf, solver.layout.hilbert_dim);

    std::cout << "Backend grouped left entry (0,27): "
              << grouped_left.at(0 * solver.layout.hilbert_dim + 27) << std::endl;

    std::cout << "Backend grouped right entry (0,27): "
              << grouped_right.at(0 * solver.layout.hilbert_dim + 27) << std::endl;

    std::cout << "Backend grouped commutator entry (0,27): "
              << grouped_comm.at(0 * solver.layout.hilbert_dim + 27) << std::endl;
    std::cout << "Dense embedded commutator entry (0,27): "
              << dense_comm.at(0 * solver.layout.hilbert_dim + 27) << std::endl;
    std::cout << "Difference backend-grouped-vs-dense commutator at (0,27): "
              << (grouped_comm.at(0 * solver.layout.hilbert_dim + 27)
                  - dense_comm.at(0 * solver.layout.hilbert_dim + 27))
              << std::endl;

    std::cout << "Backend grouped dissipator entry (0,27): "
              << grouped_diss.at(0 * solver.layout.hilbert_dim + 27) << std::endl;
    std::cout << "Dense embedded dissipator entry (0,27): "
              << dense_diss.at(0 * solver.layout.hilbert_dim + 27) << std::endl;
    std::cout << "Difference backend-grouped-vs-dense dissipator at (0,27): "
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
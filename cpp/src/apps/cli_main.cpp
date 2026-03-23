#include <complex>
#include <iostream>
#include <vector>

#include <petscts.h>

#include "culindblad/backend.hpp"
#include "culindblad/cutensor_contraction_desc.hpp"
#include "culindblad/cutensor_dissipator.hpp"
#include "culindblad/cutensor_execute.hpp"
#include "culindblad/cutensor_executor.hpp"
#include "culindblad/cutensor_operation_desc.hpp"
#include "culindblad/cutensor_ops.hpp"
#include "culindblad/cutensor_plan.hpp"
#include "culindblad/cutensor_tensor_descs.hpp"
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

    std::vector<Complex> lowering_three_site_dag(27 * 27, Complex{0.0, 0.0});
    for (Index row = 0; row < 27; ++row) {
        for (Index col = 0; col < 27; ++col) {
            lowering_three_site_dag[row * 27 + col] =
                std::conj(lowering_three_site[col * 27 + row]);
        }
    }

    std::vector<Complex> lowering_three_site_dag_op(27 * 27, Complex{0.0, 0.0});
    for (Index i = 0; i < 27; ++i) {
        for (Index j = 0; j < 27; ++j) {
            Complex accum{0.0, 0.0};
            for (Index k = 0; k < 27; ++k) {
                accum += lowering_three_site_dag[i * 27 + k]
                      * lowering_three_site[k * 27 + j];
            }
            lowering_three_site_dag_op[i * 27 + j] = accum;
        }
    }

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

    CuTensorContractionDesc cutensor_left =
        make_cutensor_left_contraction_desc({0, 1, 2}, local_dims);
    CuTensorContractionDesc cutensor_right =
        make_cutensor_right_contraction_desc({0, 1, 2}, local_dims);

    std::vector<Complex> grouped_input(27 * 3 * 27 * 3, Complex{0.0, 0.0});
    std::vector<Complex> grouped_output_left(27 * 3 * 27 * 3, Complex{0.0, 0.0});
    std::vector<Complex> grouped_output_right(27 * 3 * 27 * 3, Complex{0.0, 0.0});

    for (Index ket_flat = 0; ket_flat < solver.layout.hilbert_dim; ++ket_flat) {
        const Index ket_target = ket_flat / 3;
        const Index ket_comp = ket_flat % 3;

        for (Index bra_flat = 0; bra_flat < solver.layout.hilbert_dim; ++bra_flat) {
            const Index bra_target = bra_flat / 3;
            const Index bra_comp = bra_flat % 3;

            const Index grouped_idx =
                ((ket_target * 3 + ket_comp) * 27 + bra_target) * 3 + bra_comp;

            grouped_input[grouped_idx] =
                rho_in[ket_flat * solver.layout.hilbert_dim + bra_flat];
        }
    }

    std::cout << "cutensor left debug name: " << cutensor_left.debug_name << std::endl;
    std::cout << "cutensor right debug name: " << cutensor_right.debug_name << std::endl;

    const bool cutensor_left_valid =
        validate_cutensor_contraction_desc(cutensor_left);
    const bool cutensor_right_valid =
        validate_cutensor_contraction_desc(cutensor_right);

    std::cout << "cutensor left descriptor valid: "
              << (cutensor_left_valid ? "true" : "false") << std::endl;
    std::cout << "cutensor right descriptor valid: "
              << (cutensor_right_valid ? "true" : "false") << std::endl;

    const bool cutensor_handle_ok =
        initialize_cutensor_handle_for_desc(cutensor_left);
    std::cout << "cutensor handle initialization: "
              << (cutensor_handle_ok ? "true" : "false") << std::endl;

    CuTensorTensorDescs cutensor_tensor_descs{};
    const bool cutensor_tensor_descs_ok =
        create_cutensor_tensor_descs(cutensor_left, cutensor_tensor_descs);
    std::cout << "cutensor tensor descriptor creation: "
              << (cutensor_tensor_descs_ok ? "true" : "false") << std::endl;

    if (cutensor_tensor_descs_ok) {
        const bool cutensor_tensor_descs_destroy_ok =
            destroy_cutensor_tensor_descs(cutensor_tensor_descs);
        std::cout << "cutensor tensor descriptor destruction: "
                  << (cutensor_tensor_descs_destroy_ok ? "true" : "false") << std::endl;
    }

    CuTensorOperationDesc cutensor_op_desc{};
    const bool cutensor_op_desc_ok =
        create_cutensor_operation_desc(cutensor_left, cutensor_op_desc);
    std::cout << "cutensor operation descriptor creation: "
              << (cutensor_op_desc_ok ? "true" : "false") << std::endl;

    if (cutensor_op_desc_ok) {
        const bool cutensor_op_desc_destroy_ok =
            destroy_cutensor_operation_desc(cutensor_op_desc);
        std::cout << "cutensor operation descriptor destruction: "
                  << (cutensor_op_desc_destroy_ok ? "true" : "false") << std::endl;
    }

    CuTensorPlanBundle plan_bundle{};
    const bool plan_ok =
        create_cutensor_plan(cutensor_left, plan_bundle);
    std::cout << "cutensor plan creation: "
              << (plan_ok ? "true" : "false") << std::endl;

    if (plan_ok) {
        std::cout << "workspace size: "
                  << plan_bundle.workspace_size << std::endl;

        const bool plan_destroy_ok =
            destroy_cutensor_plan(plan_bundle);
        std::cout << "cutensor plan destruction: "
                  << (plan_destroy_ok ? "true" : "false") << std::endl;
    }

    const bool cutensor_left_exec_ok =
        execute_cutensor_left_contraction(
            cutensor_left,
            zzz_three_site,
            grouped_input,
            grouped_output_left);
    std::cout << "cutensor left execution: "
              << (cutensor_left_exec_ok ? "true" : "false") << std::endl;

    if (cutensor_left_exec_ok) {
        const Index grouped_out_idx =
            ((0 * 3 + 0) * 27 + 9) * 3 + 0;
        std::cout << "cutensor grouped left entry ((0,0),(9,0)): "
                  << grouped_output_left.at(grouped_out_idx) << std::endl;
    }

    const bool cutensor_right_exec_ok =
        execute_cutensor_right_contraction(
            cutensor_right,
            zzz_three_site,
            grouped_input,
            grouped_output_right);
    std::cout << "cutensor right execution: "
              << (cutensor_right_exec_ok ? "true" : "false") << std::endl;

    if (cutensor_right_exec_ok) {
        const Index grouped_out_idx =
            ((0 * 3 + 0) * 27 + 9) * 3 + 0;
        std::cout << "cutensor grouped right entry ((0,0),(9,0)): "
                  << grouped_output_right.at(grouped_out_idx) << std::endl;
    }

    if (cutensor_left_exec_ok && cutensor_right_exec_ok) {
        std::vector<Complex> gpu_grouped_comm(grouped_output_left.size(), Complex{0.0, 0.0});
        const Complex minus_i{0.0, -1.0};

        for (Index idx = 0; idx < gpu_grouped_comm.size(); ++idx) {
            gpu_grouped_comm[idx] =
                minus_i * (grouped_output_left[idx] - grouped_output_right[idx]);
        }

        const Index grouped_out_idx =
            ((0 * 3 + 0) * 27 + 9) * 3 + 0;
        std::cout << "cutensor grouped commutator entry ((0,0),(9,0)): "
                  << gpu_grouped_comm.at(grouped_out_idx) << std::endl;
    }

    CuTensorExecutor staged_left_executor{};
    const bool staged_left_executor_ok =
        create_cutensor_executor(
            cutensor_left,
            zzz_three_site.size() * sizeof(Complex),
            grouped_input.size() * sizeof(Complex),
            grouped_output_left.size() * sizeof(Complex),
            staged_left_executor);
    std::cout << "staged cutensor executor creation: "
              << (staged_left_executor_ok ? "true" : "false") << std::endl;

    if (staged_left_executor_ok) {
        const bool staged_upload_ok =
            upload_cutensor_executor_inputs(
                staged_left_executor,
                zzz_three_site,
                grouped_input);
        std::cout << "staged cutensor upload: "
                  << (staged_upload_ok ? "true" : "false") << std::endl;

        bool staged_exec_ok = false;
        bool staged_download_ok = false;

        if (staged_upload_ok) {
            staged_exec_ok =
                execute_cutensor_executor_device(staged_left_executor);
            std::cout << "staged cutensor device execution: "
                      << (staged_exec_ok ? "true" : "false") << std::endl;
        }

        if (staged_exec_ok) {
            staged_download_ok =
                download_cutensor_executor_output(
                    staged_left_executor,
                    grouped_output_left);
            std::cout << "staged cutensor download: "
                      << (staged_download_ok ? "true" : "false") << std::endl;
        }

        if (staged_download_ok) {
            const Index grouped_out_idx =
                ((0 * 3 + 0) * 27 + 9) * 3 + 0;
            std::cout << "staged cutensor grouped left entry ((0,0),(9,0)): "
                      << grouped_output_left.at(grouped_out_idx) << std::endl;
        }

        const bool staged_destroy_ok =
            destroy_cutensor_executor(staged_left_executor);
        std::cout << "staged cutensor executor destruction: "
                  << (staged_destroy_ok ? "true" : "false") << std::endl;
    }

    std::vector<Complex> grouped_output_diss_gpu_staged(
        27 * 3 * 27 * 3, Complex{0.0, 0.0});

    CuTensorExecutor jump_left_executor{};
    CuTensorExecutor jump_right_executor{};
    CuTensorExecutor norm_left_executor{};
    CuTensorExecutor norm_right_executor{};

    const std::size_t grouped_bytes =
        grouped_input.size() * sizeof(Complex);
    const std::size_t local_bytes =
        lowering_three_site.size() * sizeof(Complex);

    const bool jump_left_ok =
        create_cutensor_executor(
            cutensor_left,
            local_bytes,
            grouped_bytes,
            grouped_bytes,
            jump_left_executor);

    const bool jump_right_ok =
        create_cutensor_executor(
            cutensor_right,
            local_bytes,
            grouped_bytes,
            grouped_bytes,
            jump_right_executor);

    const bool norm_left_ok =
        create_cutensor_executor(
            cutensor_left,
            local_bytes,
            grouped_bytes,
            grouped_bytes,
            norm_left_executor);

    const bool norm_right_ok =
        create_cutensor_executor(
            cutensor_right,
            local_bytes,
            grouped_bytes,
            grouped_bytes,
            norm_right_executor);

    std::cout << "staged dissipator executor creation: "
              << ((jump_left_ok && jump_right_ok && norm_left_ok && norm_right_ok) ? "true" : "false")
              << std::endl;

    if (jump_left_ok && jump_right_ok && norm_left_ok && norm_right_ok) {
        const bool staged_diss_ok =
            execute_cutensor_dissipator_staged(
                jump_left_executor,
                jump_right_executor,
                norm_left_executor,
                norm_right_executor,
                lowering_three_site,
                lowering_three_site_dag,
                lowering_three_site_dag_op,
                grouped_input,
                grouped_output_diss_gpu_staged);

        std::cout << "staged cutensor dissipator execution: "
                  << (staged_diss_ok ? "true" : "false") << std::endl;

        if (staged_diss_ok) {
            const Index grouped_out_idx =
                ((0 * 3 + 0) * 27 + 9) * 3 + 0;
            std::cout << "staged cutensor grouped dissipator entry ((0,0),(9,0)): "
                      << grouped_output_diss_gpu_staged.at(grouped_out_idx) << std::endl;
        }

        const bool destroy_all_ok =
            destroy_cutensor_executor(jump_left_executor) &&
            destroy_cutensor_executor(jump_right_executor) &&
            destroy_cutensor_executor(norm_left_executor) &&
            destroy_cutensor_executor(norm_right_executor);

        std::cout << "staged dissipator executor destruction: "
                  << (destroy_all_ok ? "true" : "false") << std::endl;
    }

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
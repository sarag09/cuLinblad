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
#include "culindblad/time_dependence.hpp"
#include "culindblad/time_dependent_term.hpp"
#include "culindblad/cutensor_executor_cache.hpp"
#include "culindblad/pinned_host_buffer.hpp"
#include "culindblad/grouped_state_layout.hpp"
#include "culindblad/petsc_apply.hpp"
#include "culindblad/petsc_cuda_apply.hpp"
#include "culindblad/petsc_cuda_ts_smoke.hpp"

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

    TimeDependentTerm td_h_term_1{
        "q1_q2_q3_drive_1",
        {0, 1, 2},
        zzz_three_site,
        27,
        27,
        make_cosine_time_scalar(2.0, 3.0, 0.0)
    };

    TimeDependentTerm td_h_term_2{
        "q1_q2_q3_drive_2",
        {0, 1, 2},
        zzz_three_site,
        27,
        27,
        make_cosine_time_scalar(0.5, 5.0, 0.25)
    };

    Model model{
        local_dims,
        {h_term},
        {d_term},
        {td_h_term_1, td_h_term_2}
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

    GroupedStateLayout grouped_layout =
        make_grouped_state_layout({0, 1, 2}, local_dims);

    std::vector<Complex> grouped_input(grouped_layout.grouped_size, Complex{0.0, 0.0});
    std::vector<Complex> grouped_output_left(grouped_layout.grouped_size, Complex{0.0, 0.0});
    std::vector<Complex> grouped_output_right(grouped_layout.grouped_size, Complex{0.0, 0.0});

    regroup_flat_density_to_grouped(
        grouped_layout,
        rho_in,
        grouped_input);    

    std::vector<Complex> flat_roundtrip(rho_in.size(), Complex{0.0, 0.0});
    regroup_grouped_to_flat_density(
        grouped_layout,
        grouped_input,
        flat_roundtrip);

    std::cout << "grouped layout roundtrip difference at flat (0,27): "
              << (flat_roundtrip.at(0 * solver.layout.hilbert_dim + 27)
                  - rho_in.at(0 * solver.layout.hilbert_dim + 27))
              << std::endl;        

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

    CuTensorExecutor resident_left_executor{};
    const bool resident_left_executor_ok =
        create_cutensor_executor(
            cutensor_left,
            zzz_three_site.size() * sizeof(Complex),
            grouped_input.size() * sizeof(Complex),
            grouped_output_left.size() * sizeof(Complex),
            resident_left_executor);

    std::cout << "resident cutensor executor creation: "
              << (resident_left_executor_ok ? "true" : "false") << std::endl;

    if (resident_left_executor_ok) {
        const bool resident_upload_ok =
            upload_cutensor_executor_operator(
                resident_left_executor,
                zzz_three_site);

        std::cout << "resident cutensor operator upload: "
                  << (resident_upload_ok ? "true" : "false") << std::endl;

        std::vector<Complex> resident_output(grouped_output_left.size(), Complex{0.0, 0.0});

        bool resident_exec_ok = false;
        if (resident_upload_ok) {
            resident_exec_ok =
                execute_cutensor_executor_with_resident_operator(
                    resident_left_executor,
                    grouped_input,
                    resident_output);

            std::cout << "resident cutensor execution: "
                      << (resident_exec_ok ? "true" : "false") << std::endl;
        }

        if (resident_exec_ok) {
            const Index grouped_out_idx =
                ((0 * 3 + 0) * 27 + 9) * 3 + 0;

            std::cout << "resident cutensor grouped left entry ((0,0),(9,0)): "
                      << resident_output.at(grouped_out_idx) << std::endl;
        }

        const bool resident_destroy_ok =
            destroy_cutensor_executor(resident_left_executor);

        std::cout << "resident cutensor executor destruction: "
                  << (resident_destroy_ok ? "true" : "false") << std::endl;
    }    

    PinnedComplexBuffer pinned_input{};
    PinnedComplexBuffer pinned_output{};

    const bool pinned_input_ok =
        create_pinned_complex_buffer(grouped_input.size(), pinned_input);
    const bool pinned_output_ok =
        create_pinned_complex_buffer(grouped_output_left.size(), pinned_output);

    std::cout << "pinned input buffer creation: "
              << (pinned_input_ok ? "true" : "false") << std::endl;
    std::cout << "pinned output buffer creation: "
              << (pinned_output_ok ? "true" : "false") << std::endl;

    if (pinned_input_ok && pinned_output_ok) {
        for (Index i = 0; i < grouped_input.size(); ++i) {
            pinned_input.data[i] = grouped_input[i];
            pinned_output.data[i] = Complex{0.0, 0.0};
        }

        CuTensorExecutor pinned_executor{};
        const bool pinned_executor_ok =
            create_cutensor_executor(
                cutensor_left,
                zzz_three_site.size() * sizeof(Complex),
                grouped_input.size() * sizeof(Complex),
                grouped_output_left.size() * sizeof(Complex),
                pinned_executor);

        std::cout << "pinned cutensor executor creation: "
                  << (pinned_executor_ok ? "true" : "false") << std::endl;

        if (pinned_executor_ok) {
            const bool pinned_op_upload_ok =
                upload_cutensor_executor_operator(
                    pinned_executor,
                    zzz_three_site);

            std::cout << "pinned cutensor operator upload: "
                      << (pinned_op_upload_ok ? "true" : "false") << std::endl;

            bool pinned_exec_ok = false;
            if (pinned_op_upload_ok) {
                pinned_exec_ok =
                    execute_cutensor_executor_with_resident_operator_pinned(
                        pinned_executor,
                        pinned_input,
                        pinned_output);

                std::cout << "pinned cutensor execution: "
                          << (pinned_exec_ok ? "true" : "false") << std::endl;
            }

            if (pinned_exec_ok) {
                const Index grouped_out_idx =
                    ((0 * 3 + 0) * 27 + 9) * 3 + 0;

                std::cout << "pinned cutensor grouped left entry ((0,0),(9,0)): "
                          << pinned_output.data[grouped_out_idx] << std::endl;
            }

            const bool pinned_executor_destroy_ok =
                destroy_cutensor_executor(pinned_executor);

            std::cout << "pinned cutensor executor destruction: "
                      << (pinned_executor_destroy_ok ? "true" : "false") << std::endl;
        }
    }

    const bool pinned_input_destroy_ok =
        destroy_pinned_complex_buffer(pinned_input);
    const bool pinned_output_destroy_ok =
        destroy_pinned_complex_buffer(pinned_output);

    std::cout << "pinned input buffer destruction: "
              << (pinned_input_destroy_ok ? "true" : "false") << std::endl;
    std::cout << "pinned output buffer destruction: "
              << (pinned_output_destroy_ok ? "true" : "false") << std::endl;    

    CuTensorExecutorCache executor_cache{};

    CuTensorExecutor* cached_left_executor_1 = nullptr;
    CuTensorExecutor* cached_left_executor_2 = nullptr;

    const bool cache_create_1_ok =
        get_or_create_cutensor_executor(
            executor_cache,
            "left_zzz_q123",
            cutensor_left,
            zzz_three_site.size() * sizeof(Complex),
            grouped_input.size() * sizeof(Complex),
            grouped_output_left.size() * sizeof(Complex),
            cached_left_executor_1);

    const bool cache_create_2_ok =
        get_or_create_cutensor_executor(
            executor_cache,
            "left_zzz_q123",
            cutensor_left,
            zzz_three_site.size() * sizeof(Complex),
            grouped_input.size() * sizeof(Complex),
            grouped_output_left.size() * sizeof(Complex),
            cached_left_executor_2);

    std::cout << "cutensor executor cache first lookup: "
              << (cache_create_1_ok ? "true" : "false") << std::endl;

    std::cout << "cutensor executor cache second lookup: "
              << (cache_create_2_ok ? "true" : "false") << std::endl;

    std::cout << "cutensor executor cache reused same object: "
              << ((cached_left_executor_1 == cached_left_executor_2) ? "true" : "false")
              << std::endl;

    if (cache_create_1_ok && cached_left_executor_1 != nullptr) {
        const bool cache_operator_upload_ok =
            upload_cutensor_executor_operator(
                *cached_left_executor_1,
                zzz_three_site);

        std::cout << "cutensor executor cache operator upload: "
                  << (cache_operator_upload_ok ? "true" : "false") << std::endl;

        std::vector<Complex> cached_output(grouped_output_left.size(), Complex{0.0, 0.0});

        bool cache_exec_ok = false;
        if (cache_operator_upload_ok) {
            cache_exec_ok =
                execute_cutensor_executor_with_resident_operator(
                    *cached_left_executor_1,
                    grouped_input,
                    cached_output);

            std::cout << "cutensor executor cache execution: "
                      << (cache_exec_ok ? "true" : "false") << std::endl;
        }

        if (cache_exec_ok) {
            const Index grouped_out_idx =
                ((0 * 3 + 0) * 27 + 9) * 3 + 0;

            std::cout << "cutensor executor cache grouped left entry ((0,0),(9,0)): "
                      << cached_output.at(grouped_out_idx) << std::endl;
        }
    }

    const bool cache_destroy_ok =
        destroy_cutensor_executor_cache(executor_cache);

    std::cout << "cutensor executor cache destruction: "
              << (cache_destroy_ok ? "true" : "false") << std::endl;    

    CuTensorExecutorCache persistent_cache{};
    CuTensorExecutor* persistent_left_executor = nullptr;

    const bool persistent_lookup_ok =
        get_or_create_cutensor_executor(
            persistent_cache,
            "persistent_left_zzz_q123",
            cutensor_left,
            zzz_three_site.size() * sizeof(Complex),
            grouped_input.size() * sizeof(Complex),
            grouped_output_left.size() * sizeof(Complex),
            persistent_left_executor);

    std::cout << "persistent cached executor lookup: "
              << (persistent_lookup_ok ? "true" : "false") << std::endl;

    if (persistent_lookup_ok && persistent_left_executor != nullptr) {
        const bool ensure_first_ok =
            ensure_cutensor_executor_operator(
                *persistent_left_executor,
                "zzz_three_site_v1",
                zzz_three_site);

        std::cout << "persistent cached operator ensure first: "
                  << (ensure_first_ok ? "true" : "false") << std::endl;

        std::vector<Complex> persistent_output_1(
            grouped_output_left.size(), Complex{0.0, 0.0});

        bool persistent_exec_1_ok = false;
        if (ensure_first_ok) {
            persistent_exec_1_ok =
                execute_cutensor_executor_with_resident_operator(
                    *persistent_left_executor,
                    grouped_input,
                    persistent_output_1);

            std::cout << "persistent cached execution first: "
                      << (persistent_exec_1_ok ? "true" : "false") << std::endl;
        }

        if (persistent_exec_1_ok) {
            const Index grouped_out_idx =
                ((0 * 3 + 0) * 27 + 9) * 3 + 0;

            std::cout << "persistent cached grouped left entry first ((0,0),(9,0)): "
                      << persistent_output_1.at(grouped_out_idx) << std::endl;
        }

        const bool ensure_second_ok =
            ensure_cutensor_executor_operator(
                *persistent_left_executor,
                "zzz_three_site_v1",
                zzz_three_site);

        std::cout << "persistent cached operator ensure second same tag: "
                  << (ensure_second_ok ? "true" : "false") << std::endl;

        std::vector<Complex> persistent_output_2(
            grouped_output_left.size(), Complex{0.0, 0.0});

        bool persistent_exec_2_ok = false;
        if (ensure_second_ok) {
            persistent_exec_2_ok =
                execute_cutensor_executor_with_resident_operator(
                    *persistent_left_executor,
                    grouped_input,
                    persistent_output_2);

            std::cout << "persistent cached execution second: "
                      << (persistent_exec_2_ok ? "true" : "false") << std::endl;
        }

        if (persistent_exec_2_ok) {
            const Index grouped_out_idx =
                ((0 * 3 + 0) * 27 + 9) * 3 + 0;

            std::cout << "persistent cached grouped left entry second ((0,0),(9,0)): "
                      << persistent_output_2.at(grouped_out_idx) << std::endl;
        }
    }

    const bool persistent_cache_destroy_ok =
        destroy_cutensor_executor_cache(persistent_cache);

    std::cout << "persistent cached executor destruction: "
              << (persistent_cache_destroy_ok ? "true" : "false") << std::endl;              

    std::vector<Complex> grouped_output_diss_gpu_staged(
        grouped_layout.grouped_size, Complex{0.0, 0.0});

    CuTensorExecutorCache dissipator_executor_cache{};

    CuTensorExecutor* jump_left_executor = nullptr;
    CuTensorExecutor* jump_right_executor = nullptr;
    CuTensorExecutor* norm_left_executor = nullptr;
    CuTensorExecutor* norm_right_executor = nullptr;

    const std::size_t grouped_bytes =
        grouped_input.size() * sizeof(Complex);
    const std::size_t local_bytes =
        lowering_three_site.size() * sizeof(Complex);

    const bool jump_left_ok =
        get_or_create_cutensor_executor(
            dissipator_executor_cache,
            "jump_left_q123",
            cutensor_left,
            local_bytes,
            grouped_bytes,
            grouped_bytes,
            jump_left_executor);

    const bool jump_right_ok =
        get_or_create_cutensor_executor(
            dissipator_executor_cache,
            "jump_right_q123",
            cutensor_right,
            local_bytes,
            grouped_bytes,
            grouped_bytes,
            jump_right_executor);

    const bool norm_left_ok =
        get_or_create_cutensor_executor(
            dissipator_executor_cache,
            "norm_left_q123",
            cutensor_left,
            local_bytes,
            grouped_bytes,
            grouped_bytes,
            norm_left_executor);

    const bool norm_right_ok =
        get_or_create_cutensor_executor(
            dissipator_executor_cache,
            "norm_right_q123",
            cutensor_right,
            local_bytes,
            grouped_bytes,
            grouped_bytes,
            norm_right_executor);

    std::cout << "cached staged dissipator executor lookup: "
              << ((jump_left_ok && jump_right_ok && norm_left_ok && norm_right_ok) ? "true" : "false")
              << std::endl;

    bool cached_operator_uploads_ok = false;
    if (jump_left_ok && jump_right_ok && norm_left_ok && norm_right_ok) {
        const bool jump_left_upload_ok =
            upload_cutensor_executor_operator(*jump_left_executor, lowering_three_site);
        const bool jump_right_upload_ok =
            upload_cutensor_executor_operator(*jump_right_executor, lowering_three_site_dag);
        const bool norm_left_upload_ok =
            upload_cutensor_executor_operator(*norm_left_executor, lowering_three_site_dag_op);
        const bool norm_right_upload_ok =
            upload_cutensor_executor_operator(*norm_right_executor, lowering_three_site_dag_op);

        cached_operator_uploads_ok =
            jump_left_upload_ok &&
            jump_right_upload_ok &&
            norm_left_upload_ok &&
            norm_right_upload_ok;

        std::cout << "cached staged dissipator operator uploads: "
                  << (cached_operator_uploads_ok ? "true" : "false") << std::endl;
    }

    if (cached_operator_uploads_ok) {
        const bool staged_diss_ok =
            execute_cutensor_dissipator_staged(
                *jump_left_executor,
                *jump_right_executor,
                *norm_left_executor,
                *norm_right_executor,
                lowering_three_site,
                lowering_three_site_dag,
                lowering_three_site_dag_op,
                grouped_input,
                grouped_output_diss_gpu_staged);

        std::cout << "cached staged cutensor dissipator execution: "
                  << (staged_diss_ok ? "true" : "false") << std::endl;

        if (staged_diss_ok) {
            const Index grouped_out_idx =
                ((0 * 3 + 0) * 27 + 9) * 3 + 0;

            std::cout << "cached staged cutensor grouped dissipator entry ((0,0),(9,0)): "
                      << grouped_output_diss_gpu_staged.at(grouped_out_idx) << std::endl;
        }
    }

    const bool dissipator_cache_destroy_ok =
        destroy_cutensor_executor_cache(dissipator_executor_cache);

    std::cout << "cached staged dissipator executor destruction: "
              << (dissipator_cache_destroy_ok ? "true" : "false") << std::endl;

    CuTensorExecutorCache reuse_check_cache{};
    CuTensorExecutor* reuse_a = nullptr;
    CuTensorExecutor* reuse_b = nullptr;

    const bool reuse_a_ok =
        get_or_create_cutensor_executor(
            reuse_check_cache,
            "reuse_left_q123",
            cutensor_left,
            zzz_three_site.size() * sizeof(Complex),
            grouped_input.size() * sizeof(Complex),
            grouped_output_left.size() * sizeof(Complex),
            reuse_a);

    const bool reuse_b_ok =
        get_or_create_cutensor_executor(
            reuse_check_cache,
            "reuse_left_q123",
            cutensor_left,
            zzz_three_site.size() * sizeof(Complex),
            grouped_input.size() * sizeof(Complex),
            grouped_output_left.size() * sizeof(Complex),
            reuse_b);

    std::cout << "reuse-check cache same executor: "
              << ((reuse_a_ok && reuse_b_ok && reuse_a == reuse_b) ? "true" : "false")
              << std::endl;

    const bool reuse_check_cache_destroy_ok =
        destroy_cutensor_executor_cache(reuse_check_cache);

    std::cout << "reuse-check cache destruction: "
              << (reuse_check_cache_destroy_ok ? "true" : "false") << std::endl;              

    TimeDependentTerm td_drive{
        "q1_q2_q3_drive",
        {0, 1, 2},
        zzz_three_site,
        27,
        27,
        make_cosine_time_scalar(2.0, 3.0, 0.0)
    };

    std::vector<Complex> rho_out_timed(solver.layout.density_dim, Complex{0.0, 0.0});
    StateBuffer out_buf_timed{rho_out_timed.data(), rho_out_timed.size()};

    const double timed_test_t = 0.5;
    apply_liouvillian_at_time(solver, timed_test_t, in_buf, out_buf_timed);

    std::cout << "time-dependent coefficient 1 at t=0.5: "
              << evaluate_time_scalar(td_h_term_1.coefficient, timed_test_t) << std::endl;
    std::cout << "time-dependent coefficient 2 at t=0.5: "
              << evaluate_time_scalar(td_h_term_2.coefficient, timed_test_t) << std::endl;
    std::cout << "Timed backend k-site total entry (0,27): "
              << rho_out_timed.at(0 * solver.layout.hilbert_dim + 27) << std::endl; 
              
    Complex petsc_cuda_value{0.0, 0.0};
    ierr = petsc_cuda_vec_smoke_test(solver, 0, 27, petsc_cuda_value);
    if (ierr != 0) {
        std::cerr << "petsc_cuda_vec_smoke_test failed." << std::endl;
        PetscFinalize();
        return 1;
    }

    std::cout << "PETSc VECCUDA smoke entry (0,27): "
              << petsc_cuda_value << std::endl;       
              
    Vec x_cuda = nullptr;
    Vec y_cuda = nullptr;

    PetscCall(VecCreate(PETSC_COMM_SELF, &x_cuda));
    PetscCall(VecSetSizes(x_cuda, PETSC_DECIDE, solver.layout.density_dim));
    PetscCall(VecSetType(x_cuda, VECCUDA));
    PetscCall(VecDuplicate(x_cuda, &y_cuda));
    PetscCall(VecSet(x_cuda, 0.0));
    PetscCall(VecSet(y_cuda, 0.0));

    PetscScalar* x_cuda_ptr = nullptr;
    PetscCall(VecGetArray(x_cuda, &x_cuda_ptr));
    x_cuda_ptr[0 * solver.layout.hilbert_dim + 27] = PetscScalar(1.0);
    PetscCall(VecRestoreArray(x_cuda, &x_cuda_ptr));

    CuTensorExecutorCache petsc_cuda_cache{};

    PetscCall(apply_grouped_left_cuda_vec(
        solver,
        zzz_three_site,
        {0, 1, 2},
        grouped_layout,
        petsc_cuda_cache,
        x_cuda,
        y_cuda));

    PetscScalar* y_cuda_ptr = nullptr;
    PetscCall(VecGetArray(y_cuda, &y_cuda_ptr));

    std::cout << "PETSc VECCUDA grouped-left entry (0,27): "
              << reinterpret_cast<Complex*>(y_cuda_ptr)[0 * solver.layout.hilbert_dim + 27]
              << std::endl;

    PetscCall(VecRestoreArray(y_cuda, &y_cuda_ptr));

    const bool petsc_cuda_cache_destroy_ok =
        destroy_cutensor_executor_cache(petsc_cuda_cache);

    std::cout << "PETSc VECCUDA grouped-left cache destruction: "
              << (petsc_cuda_cache_destroy_ok ? "true" : "false") << std::endl;

    PetscCall(VecDestroy(&y_cuda));
    PetscCall(VecDestroy(&x_cuda));       
    
    std::cout << "Running PETSc VECCUDA grouped-left TS smoke test" << std::endl;

    Complex ts_cuda_value{0.0, 0.0};
    PetscCall(run_ts_cuda_grouped_left_smoke_test(
        solver,
        zzz_three_site,
        {0, 1, 2},
        0,
        27,
        ts_cuda_value));

    std::cout << "PETSc VECCUDA grouped-left TS entry (0,27): "
              << ts_cuda_value << std::endl;   
              
    Vec x_cuda_diss = nullptr;
    Vec y_cuda_diss = nullptr;

    PetscCall(VecCreate(PETSC_COMM_SELF, &x_cuda_diss));
    PetscCall(VecSetSizes(x_cuda_diss, PETSC_DECIDE, solver.layout.density_dim));
    PetscCall(VecSetType(x_cuda_diss, VECCUDA));
    PetscCall(VecDuplicate(x_cuda_diss, &y_cuda_diss));
    PetscCall(VecSet(x_cuda_diss, 0.0));
    PetscCall(VecSet(y_cuda_diss, 0.0));

    PetscScalar* x_cuda_diss_ptr = nullptr;
    PetscCall(VecGetArray(x_cuda_diss, &x_cuda_diss_ptr));
    x_cuda_diss_ptr[0 * solver.layout.hilbert_dim + 27] = PetscScalar(1.0);
    PetscCall(VecRestoreArray(x_cuda_diss, &x_cuda_diss_ptr));

    CuTensorExecutorCache petsc_cuda_diss_cache{};

    PetscCall(apply_grouped_dissipator_cuda_vec(
        solver,
        lowering_three_site,
        lowering_three_site_dag,
        lowering_three_site_dag_op,
        {0, 1, 2},
        grouped_layout,
        petsc_cuda_diss_cache,
        x_cuda_diss,
        y_cuda_diss));

    PetscScalar* y_cuda_diss_ptr = nullptr;
    PetscCall(VecGetArray(y_cuda_diss, &y_cuda_diss_ptr));

    std::cout << "PETSc VECCUDA grouped dissipator entry (0,27): "
              << reinterpret_cast<Complex*>(y_cuda_diss_ptr)[0 * solver.layout.hilbert_dim + 27]
              << std::endl;

    PetscCall(VecRestoreArray(y_cuda_diss, &y_cuda_diss_ptr));

    const bool petsc_cuda_diss_cache_destroy_ok =
        destroy_cutensor_executor_cache(petsc_cuda_diss_cache);

    std::cout << "PETSc VECCUDA grouped dissipator cache destruction: "
              << (petsc_cuda_diss_cache_destroy_ok ? "true" : "false") << std::endl;

    PetscCall(VecDestroy(&y_cuda_diss));
    PetscCall(VecDestroy(&x_cuda_diss));              
              
    std::cout << "Running TS smoke test with time-dependent RHS" << std::endl;              
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
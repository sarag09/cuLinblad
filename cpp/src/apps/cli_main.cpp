#include <complex>
#include <iostream>
#include <vector>

#include <petscts.h>

#include "culindblad/backend.hpp"
#include "culindblad/cuda_grouped_layout.hpp"
#include "culindblad/cutensor_contraction_desc.hpp"
#include "culindblad/cutensor_execute.hpp"
#include "culindblad/cutensor_ops.hpp"
#include "culindblad/cutensor_plan.hpp"
#include "culindblad/cutensor_tensor_descs.hpp"
#include "culindblad/grouped_contraction_backend.hpp"
#include "culindblad/grouped_state_layout.hpp"
#include "culindblad/k_site_operator_embed.hpp"
#include "culindblad/liouvillian_terms.hpp"
#include "culindblad/model.hpp"
#include "culindblad/operator_term.hpp"
#include "culindblad/petsc_apply.hpp"
#include "culindblad/petsc_cuda_apply.hpp"
#include "culindblad/petsc_cuda_ts_smoke.hpp"
#include "culindblad/petsc_ts_smoke.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/time_dependence.hpp"
#include "culindblad/time_dependent_term.hpp"
#include "culindblad/types.hpp"
#include "culindblad/petsc_cuda_vec_utils.hpp"

int main(int argc, char** argv)
{
    using namespace culindblad;

    PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
    if (ierr != 0) {
        std::cerr << "PetscInitialize failed." << std::endl;
        return 1;
    }

    const std::vector<Index> local_dims = {3, 3, 3, 3};
    const std::vector<Index> target_sites = {0, 1, 2};

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
        target_sites,
        zzz_three_site,
        27,
        27
    };

    OperatorTerm d_term{
        TermKind::Dissipator,
        "q1_q2_q3_jump",
        target_sites,
        lowering_three_site,
        27,
        27
    };

    TimeDependentTerm td_h_term_1{
        "q1_q2_q3_drive_1",
        target_sites,
        zzz_three_site,
        27,
        27,
        make_cosine_time_scalar(2.0, 3.0, 0.0)
    };

    TimeDependentTerm td_h_term_2{
        "q1_q2_q3_drive_2",
        target_sites,
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

    std::cout << "cuLindblad smoke test" << std::endl;
    std::cout << "Hilbert dimension: " << solver.layout.hilbert_dim << std::endl;
    std::cout << "Density dimension: " << solver.layout.density_dim << std::endl;

    std::vector<Complex> rho_in(solver.layout.density_dim, Complex{0.0, 0.0});
    std::vector<Complex> rho_out(solver.layout.density_dim, Complex{0.0, 0.0});
    rho_in[0 * solver.layout.hilbert_dim + 27] = Complex{1.0, 0.0};

    ConstStateBuffer in_buf{rho_in.data(), rho_in.size()};
    StateBuffer out_buf{rho_out.data(), rho_out.size()};

    std::vector<Complex> grouped_comm =
        apply_grouped_commutator(zzz_three_site, target_sites, local_dims, in_buf);
    std::vector<Complex> grouped_diss =
        apply_grouped_dissipator(lowering_three_site, target_sites, local_dims, in_buf);

    std::vector<Complex> dense_h =
        embed_k_site_operator(zzz_three_site, target_sites, local_dims);
    std::vector<Complex> dense_comm =
        apply_hamiltonian_commutator(dense_h, in_buf, solver.layout.hilbert_dim);

    std::vector<Complex> dense_l =
        embed_k_site_operator(lowering_three_site, target_sites, local_dims);
    std::vector<Complex> dense_diss =
        apply_dissipator(dense_l, in_buf, solver.layout.hilbert_dim);

    std::cout << "Grouped commutator entry (0,27): "
              << grouped_comm.at(0 * solver.layout.hilbert_dim + 27) << std::endl;
    std::cout << "Dense commutator entry (0,27): "
              << dense_comm.at(0 * solver.layout.hilbert_dim + 27) << std::endl;
    std::cout << "Grouped dissipator entry (0,27): "
              << grouped_diss.at(0 * solver.layout.hilbert_dim + 27) << std::endl;
    std::cout << "Dense dissipator entry (0,27): "
              << dense_diss.at(0 * solver.layout.hilbert_dim + 27) << std::endl;

    apply_liouvillian(solver, in_buf, out_buf);
    std::cout << "Backend Liouvillian entry (0,27): "
              << rho_out.at(0 * solver.layout.hilbert_dim + 27) << std::endl;

    GroupedStateLayout grouped_layout =
        make_grouped_state_layout(target_sites, local_dims);

    std::vector<Complex> grouped_input(grouped_layout.grouped_size, Complex{0.0, 0.0});
    regroup_flat_density_to_grouped(grouped_layout, rho_in, grouped_input);

    std::vector<Complex> flat_roundtrip(rho_in.size(), Complex{0.0, 0.0});
    regroup_grouped_to_flat_density(grouped_layout, grouped_input, flat_roundtrip);

    std::cout << "Host grouped roundtrip difference at flat (0,27): "
              << (flat_roundtrip.at(0 * solver.layout.hilbert_dim + 27)
                  - rho_in.at(0 * solver.layout.hilbert_dim + 27))
              << std::endl;

    CudaGroupedStateLayout cuda_grouped_layout{};
    const bool cuda_grouped_layout_ok =
        create_cuda_grouped_state_layout(grouped_layout, cuda_grouped_layout);

    std::cout << "CUDA grouped layout creation: "
              << (cuda_grouped_layout_ok ? "true" : "false") << std::endl;

    if (cuda_grouped_layout_ok) {
        void* d_flat = nullptr;
        void* d_grouped = nullptr;
        void* d_flat_roundtrip = nullptr;

        const std::size_t flat_bytes = rho_in.size() * sizeof(Complex);
        const std::size_t grouped_bytes = grouped_layout.grouped_size * sizeof(Complex);

        const bool d_flat_ok = cudaMalloc(&d_flat, flat_bytes) == cudaSuccess;
        const bool d_grouped_ok = cudaMalloc(&d_grouped, grouped_bytes) == cudaSuccess;
        const bool d_flat_roundtrip_ok = cudaMalloc(&d_flat_roundtrip, flat_bytes) == cudaSuccess;

        if (d_flat_ok && d_grouped_ok && d_flat_roundtrip_ok) {
            const bool h2d_ok =
                cudaMemcpy(d_flat, rho_in.data(), flat_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
            const bool flat_to_grouped_ok =
                h2d_ok && launch_flat_to_grouped_kernel(cuda_grouped_layout, d_flat, d_grouped, 0);
            const bool grouped_to_flat_ok =
                flat_to_grouped_ok &&
                launch_grouped_to_flat_kernel(cuda_grouped_layout, d_grouped, d_flat_roundtrip, 0);
            const bool sync_ok =
                grouped_to_flat_ok && (cudaDeviceSynchronize() == cudaSuccess);

            std::cout << "CUDA regroup kernels: "
                      << ((h2d_ok && flat_to_grouped_ok && grouped_to_flat_ok && sync_ok) ? "true" : "false")
                      << std::endl;

            if (sync_ok) {
                std::vector<Complex> rho_roundtrip_cuda(rho_in.size(), Complex{0.0, 0.0});
                const bool d2h_ok =
                    cudaMemcpy(
                        rho_roundtrip_cuda.data(),
                        d_flat_roundtrip,
                        flat_bytes,
                        cudaMemcpyDeviceToHost) == cudaSuccess;

                if (d2h_ok) {
                    std::cout << "CUDA grouped roundtrip difference at flat (0,27): "
                              << (rho_roundtrip_cuda.at(0 * solver.layout.hilbert_dim + 27)
                                  - rho_in.at(0 * solver.layout.hilbert_dim + 27))
                              << std::endl;
                }
            }
        }

        if (d_flat_roundtrip != nullptr) {
            cudaFree(d_flat_roundtrip);
        }
        if (d_grouped != nullptr) {
            cudaFree(d_grouped);
        }
        if (d_flat != nullptr) {
            cudaFree(d_flat);
        }
    }

    CuTensorContractionDesc cutensor_left =
        make_cutensor_left_contraction_desc(target_sites, local_dims);
    CuTensorContractionDesc cutensor_right =
        make_cutensor_right_contraction_desc(target_sites, local_dims);

    std::vector<Complex> grouped_output_left(grouped_layout.grouped_size, Complex{0.0, 0.0});
    std::vector<Complex> grouped_output_right(grouped_layout.grouped_size, Complex{0.0, 0.0});

    const bool cutensor_left_exec_ok =
        execute_cutensor_left_contraction(
            cutensor_left,
            zzz_three_site,
            grouped_input,
            grouped_output_left);

    const bool cutensor_right_exec_ok =
        execute_cutensor_right_contraction(
            cutensor_right,
            zzz_three_site,
            grouped_input,
            grouped_output_right);

    std::cout << "cuTENSOR left execution: "
              << (cutensor_left_exec_ok ? "true" : "false") << std::endl;
    std::cout << "cuTENSOR right execution: "
              << (cutensor_right_exec_ok ? "true" : "false") << std::endl;

    if (cutensor_left_exec_ok) {
        const Index grouped_out_idx = ((0 * 3 + 0) * 27 + 9) * 3 + 0;
        std::cout << "cuTENSOR grouped left entry ((0,0),(9,0)): "
                  << grouped_output_left.at(grouped_out_idx) << std::endl;
    }

    if (cutensor_right_exec_ok) {
        const Index grouped_out_idx = ((0 * 3 + 0) * 27 + 9) * 3 + 0;
        std::cout << "cuTENSOR grouped right entry ((0,0),(9,0)): "
                  << grouped_output_right.at(grouped_out_idx) << std::endl;
    }

    std::vector<Complex> rho_out_timed(solver.layout.density_dim, Complex{0.0, 0.0});
    StateBuffer out_buf_timed{rho_out_timed.data(), rho_out_timed.size()};

    const double timed_test_t = 0.5;
    apply_liouvillian_at_time(solver, timed_test_t, in_buf, out_buf_timed);

    std::cout << "time-dependent coefficient 1 at t=0.5: "
              << evaluate_time_scalar(td_h_term_1.coefficient, timed_test_t) << std::endl;
    std::cout << "time-dependent coefficient 2 at t=0.5: "
              << evaluate_time_scalar(td_h_term_2.coefficient, timed_test_t) << std::endl;
    std::cout << "Timed backend Liouvillian entry (0,27): "
              << rho_out_timed.at(0 * solver.layout.hilbert_dim + 27) << std::endl;

    Complex petsc_cuda_value{0.0, 0.0};
    PetscCall(petsc_cuda_vec_smoke_test(solver, 0, 27, petsc_cuda_value));
    std::cout << "PETSc VECCUDA smoke entry (0,27): "
              << petsc_cuda_value << std::endl;

    Vec x_cuda_probe = nullptr;
    PetscCall(VecCreate(PETSC_COMM_SELF, &x_cuda_probe));
    PetscCall(VecSetSizes(x_cuda_probe, PETSC_DECIDE, solver.layout.density_dim));
    PetscCall(VecSetType(x_cuda_probe, VECCUDA));
    PetscCall(VecSet(x_cuda_probe, 0.0));

    PetscCall(petsc_cuda_vec_device_access_smoke(x_cuda_probe));
    std::cout << "PETSc VECCUDA device access smoke: true" << std::endl;

    PetscCall(VecDestroy(&x_cuda_probe));              

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
        "cli_smoke_h",
        zzz_three_site,
        {0, 1, 2},
        grouped_layout,
        cuda_grouped_layout,
        petsc_cuda_cache,
        x_cuda,
        y_cuda));

    PetscScalar* y_cuda_ptr = nullptr;
    PetscCall(VecGetArray(y_cuda, &y_cuda_ptr));
    std::cout << "PETSc VECCUDA grouped-left entry (0,27): "
              << reinterpret_cast<Complex*>(y_cuda_ptr)[0 * solver.layout.hilbert_dim + 27]
              << std::endl;
    PetscCall(VecRestoreArray(y_cuda, &y_cuda_ptr));

    std::cout << "PETSc VECCUDA grouped-left cache destruction: "
              << (destroy_cutensor_executor_cache(petsc_cuda_cache) ? "true" : "false")
              << std::endl;

    PetscCall(VecDestroy(&y_cuda));
    PetscCall(VecDestroy(&x_cuda));

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
        "cli_smoke_d",
        lowering_three_site,
        lowering_three_site_dag,
        lowering_three_site_dag_op,
        {0, 1, 2},
        grouped_layout,
        cuda_grouped_layout,
        petsc_cuda_diss_cache,
        x_cuda_diss,
        y_cuda_diss));

    PetscScalar* y_cuda_diss_ptr = nullptr;
    PetscCall(VecGetArray(y_cuda_diss, &y_cuda_diss_ptr));
    std::cout << "PETSc VECCUDA grouped dissipator entry (0,27): "
              << reinterpret_cast<Complex*>(y_cuda_diss_ptr)[0 * solver.layout.hilbert_dim + 27]
              << std::endl;
    PetscCall(VecRestoreArray(y_cuda_diss, &y_cuda_diss_ptr));

    std::cout << "PETSc VECCUDA grouped dissipator cache destruction: "
              << (destroy_cutensor_executor_cache(petsc_cuda_diss_cache) ? "true" : "false")
              << std::endl;

    PetscCall(VecDestroy(&y_cuda_diss));
    PetscCall(VecDestroy(&x_cuda_diss));

    std::cout << "Running PETSc VECCUDA grouped-left TS smoke test" << std::endl;
    Complex ts_cuda_value{0.0, 0.0};
    PetscCall(run_ts_cuda_grouped_left_smoke_test(
        solver,
        zzz_three_site,
        target_sites,
        0,
        27,
        ts_cuda_value));
    std::cout << "PETSc VECCUDA grouped-left TS entry (0,27): "
              << ts_cuda_value << std::endl;

    std::cout << "Running PETSc VECCUDA grouped Liouvillian TS smoke test" << std::endl;
    Complex ts_cuda_liouvillian_value{0.0, 0.0};
    PetscCall(run_ts_cuda_grouped_liouvillian_smoke_test(
        solver,
        zzz_three_site,
        lowering_three_site,
        lowering_three_site_dag,
        lowering_three_site_dag_op,
        target_sites,
        0,
        27,
        ts_cuda_liouvillian_value));
    std::cout << "PETSc VECCUDA grouped Liouvillian TS entry (0,27): "
              << ts_cuda_liouvillian_value << std::endl;

    std::cout << "Running PETSc VECCUDA static-model Liouvillian TS smoke test" << std::endl;

    Complex ts_cuda_static_model_value{0.0, 0.0};
    PetscCall(run_ts_cuda_static_model_liouvillian_smoke_test(
        solver,
        0,
        27,
        ts_cuda_static_model_value));

    std::cout << "PETSc VECCUDA static-model Liouvillian TS entry (0,27): "
              << ts_cuda_static_model_value << std::endl;              

    std::cout << "Running TS smoke test with time-dependent RHS" << std::endl;
    Complex ts_value{0.0, 0.0};
    ierr = run_ts_smoke_test(solver, 0, 27, ts_value);
    if (ierr != 0) {
        std::cerr << "run_ts_smoke_test failed." << std::endl;
        PetscFinalize();
        return 1;
    }

    std::cout << "TS evolved entry (0,27): " << ts_value << std::endl;

    std::cout << "Running PETSc VECCUDA full-model Liouvillian TS smoke test" << std::endl;

    Complex ts_cuda_full_model_value{0.0, 0.0};
    PetscCall(run_ts_cuda_full_model_liouvillian_smoke_test(
            solver,
            0.0,  // <--- The new time argument (t = 0.0)
            0,    // <--- row
            27,   // <--- col
            ts_cuda_full_model_value));

    std::cout << "PETSc VECCUDA full-model Liouvillian TS entry (0,27): "
              << ts_cuda_full_model_value << std::endl;    

    if (cuda_grouped_layout_ok) {
        std::cout << "CUDA grouped layout destruction: "
                  << (destroy_cuda_grouped_state_layout(cuda_grouped_layout) ? "true" : "false")
                  << std::endl;
    }

   std::cout << "\n===== Mixed-site pattern test =====\n" << std::endl; 
    // Two-site Z ⊗ Z on {0,1}
    std::vector<Complex> zz_2site(9 * 9, Complex{0.0, 0.0});
    for (Index a = 0; a < 3; ++a) {
        for (Index b = 0; b < 3; ++b) {
            const double za = (a == 0 ? 1.0 : (a == 1 ? -1.0 : 0.0));
            const double zb = (b == 0 ? 1.0 : (b == 1 ? -1.0 : 0.0));
            const Index idx = a * 3 + b;
            zz_2site[idx * 9 + idx] = Complex{za * zb, 0.0};
        }
    }

    // Two-site operator on {2,3}
    std::vector<Complex> zz_2site_23 = zz_2site;

    // Dissipator on {1,3}
    std::vector<Complex> jump_2site(9 * 9, Complex{0.0, 0.0});
    jump_2site[0 * 9 + 3] = Complex{1.0, 0.0};

    OperatorTerm h_term_01{
        TermKind::Hamiltonian,
        "zz_q0_q1",
        {0, 1},
        zz_2site,
        9,
        9
    };

    OperatorTerm h_term_23{
        TermKind::Hamiltonian,
        "zz_q2_q3",
        {2, 3},
        zz_2site_23,
        9,
        9
    };

    OperatorTerm d_term_13{
        TermKind::Dissipator,
        "jump_q1_q3",
        {1, 3},
        jump_2site,
        9,
        9
    };

    TimeDependentTerm td_term_02{
        "drive_q0_q2",
        {0, 2},
        zz_2site,
        9,
        9,
        make_cosine_time_scalar(1.0, 2.0, 0.0)
    };

    Model mixed_model{
        local_dims,
        {h_term_01, h_term_23},
        {d_term_13},
        {td_term_02}
    };

    Solver mixed_solver = make_solver(mixed_model);

    std::vector<Complex> rho_mixed_out_cpu(
        mixed_solver.layout.density_dim, Complex{0.0, 0.0});

    StateBuffer mixed_out_buf_cpu{
        rho_mixed_out_cpu.data(),
        rho_mixed_out_cpu.size()
    };

    apply_liouvillian_at_time(
        mixed_solver,
        0.5,
        in_buf,
        mixed_out_buf_cpu);

    std::cout << "Mixed CPU Liouvillian entry (0,27): "
            << rho_mixed_out_cpu.at(0 * mixed_solver.layout.hilbert_dim + 27)
            << std::endl;    

    std::cout << "Running mixed-site GPU TS test" << std::endl;

    Complex mixed_ts_value{0.0, 0.0};

    PetscCall(run_ts_cuda_full_model_liouvillian_smoke_test(
        mixed_solver,
        0.5,
        0,
        27,
        mixed_ts_value));

        std::cout << "Mixed GPU TS entry (0,27): "
                << mixed_ts_value << std::endl;            

    PetscFinalize();
    return 0;
}
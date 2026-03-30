#include <complex>
#include <iostream>
#include <vector>

#include <petscts.h>

#include "culindblad/backend.hpp"
#include "culindblad/batch_evolution.hpp"
#include "culindblad/batched_grouped_apply.hpp"
#include "culindblad/batched_grouped_layout.hpp"
#include "culindblad/cuda_grouped_layout.hpp"
#include "culindblad/grouped_contraction_backend.hpp"
#include "culindblad/grouped_state_layout.hpp"
#include "culindblad/k_site_operator_embed.hpp"
#include "culindblad/liouvillian_terms.hpp"
#include "culindblad/model.hpp"
#include "culindblad/operator_term.hpp"
#include "culindblad/petsc_cuda_apply.hpp"
#include "culindblad/petsc_cuda_ts_smoke.hpp"
#include "culindblad/petsc_cuda_vec_utils.hpp"
#include "culindblad/petsc_ts_smoke.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/time_dependence.hpp"
#include "culindblad/time_dependent_term.hpp"
#include "culindblad/types.hpp"
#include "culindblad/petsc_apply.hpp"

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

    auto make_zzz_three_site = []() {
        std::vector<Complex> op(27 * 27, Complex{0.0, 0.0});
        for (Index a = 0; a < 3; ++a) {
            for (Index b = 0; b < 3; ++b) {
                for (Index c = 0; c < 3; ++c) {
                    const double za = (a == 0 ? 1.0 : (a == 1 ? -1.0 : 0.0));
                    const double zb = (b == 0 ? 1.0 : (b == 1 ? -1.0 : 0.0));
                    const double zc = (c == 0 ? 1.0 : (c == 1 ? -1.0 : 0.0));
                    const Index idx = a * 9 + b * 3 + c;
                    op[idx * 27 + idx] = Complex{za * zb * zc, 0.0};
                }
            }
        }
        return op;
    };

    auto make_jump_three_site = []() {
        std::vector<Complex> op(27 * 27, Complex{0.0, 0.0});
        op[0 * 27 + 9] = Complex{1.0, 0.0};
        return op;
    };

    auto dagger_square = [](const std::vector<Complex>& op) {
        std::vector<Complex> op_dag(27 * 27, Complex{0.0, 0.0});
        for (Index row = 0; row < 27; ++row) {
            for (Index col = 0; col < 27; ++col) {
                op_dag[row * 27 + col] = std::conj(op[col * 27 + row]);
            }
        }
        return op_dag;
    };

    auto gram_square = [](const std::vector<Complex>& op_dag,
                          const std::vector<Complex>& op) {
        std::vector<Complex> gram(27 * 27, Complex{0.0, 0.0});
        for (Index i = 0; i < 27; ++i) {
            for (Index j = 0; j < 27; ++j) {
                Complex accum{0.0, 0.0};
                for (Index k = 0; k < 27; ++k) {
                    accum += op_dag[i * 27 + k] * op[k * 27 + j];
                }
                gram[i * 27 + j] = accum;
            }
        }
        return gram;
    };

    const std::vector<Complex> zzz_three_site = make_zzz_three_site();
    const std::vector<Complex> lowering_three_site = make_jump_three_site();
    const std::vector<Complex> lowering_three_site_dag = dagger_square(lowering_three_site);
    const std::vector<Complex> lowering_three_site_dag_op =
        gram_square(lowering_three_site_dag, lowering_three_site);

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
        target_sites,
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
        target_sites,
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
    PetscCall(run_ts_smoke_test(solver, 0, 27, ts_value));
    std::cout << "TS evolved entry (0,27): " << ts_value << std::endl;

    std::cout << "Running PETSc VECCUDA full-model Liouvillian TS smoke test" << std::endl;
    Complex ts_cuda_full_model_value{0.0, 0.0};
    PetscCall(run_ts_cuda_full_model_liouvillian_smoke_test(
        solver,
        0.0,
        0,
        27,
        ts_cuda_full_model_value));
    std::cout << "PETSc VECCUDA full-model Liouvillian TS entry (0,27): "
              << ts_cuda_full_model_value << std::endl;

    if (cuda_grouped_layout_ok) {
        std::cout << "CUDA grouped layout destruction: "
                  << (destroy_cuda_grouped_state_layout(cuda_grouped_layout) ? "true" : "false")
                  << std::endl;
    }

    std::cout << "\n===== Batch evolution smoke test =====\n" << std::endl;

    std::vector<std::vector<Complex>> batch_initial_states(
        4,
        std::vector<Complex>(solver.layout.density_dim, Complex{0.0, 0.0}));

    batch_initial_states[0][0 * solver.layout.hilbert_dim + 0] = Complex{1.0, 0.0};
    batch_initial_states[1][0 * solver.layout.hilbert_dim + 27] = Complex{1.0, 0.0};
    batch_initial_states[2][27 * solver.layout.hilbert_dim + 0] = Complex{1.0, 0.0};
    batch_initial_states[3][27 * solver.layout.hilbert_dim + 27] = Complex{1.0, 0.0};

    BatchEvolutionConfig batch_config{
        0.0,
        1.0e-3,
        1,
        2
    };

    BatchEvolutionResult batch_result =
        evolve_density_batch_cpu_reference(
            solver,
            batch_initial_states,
            batch_config);

    std::cout << "Batch result state 0 entry (0,0): "
              << batch_result.final_states[0].at(0 * solver.layout.hilbert_dim + 0)
              << std::endl;

    std::cout << "Batch result state 1 entry (0,27): "
              << batch_result.final_states[1].at(0 * solver.layout.hilbert_dim + 27)
              << std::endl;

    BatchEvolutionResult batch_result_cuda =
        evolve_density_batch_cuda_ts(
            solver,
            batch_initial_states,
            batch_config);

    std::cout << "CUDA batch result state 0 entry (0,0): "
              << batch_result_cuda.final_states[0].at(0 * solver.layout.hilbert_dim + 0)
              << std::endl;

    std::cout << "CUDA batch result state 1 entry (0,27): "
              << batch_result_cuda.final_states[1].at(0 * solver.layout.hilbert_dim + 27)
              << std::endl;

    std::cout << "\n===== Batch timing smoke test =====\n" << std::endl;

    auto make_batch_initial_states =
        [&](const Solver& active_solver, Index count) {
            std::vector<std::vector<Complex>> states(
                count,
                std::vector<Complex>(active_solver.layout.density_dim, Complex{0.0, 0.0}));

            for (Index i = 0; i < count; ++i) {
                const Index ket = (i % 2 == 0) ? 0 : 27;
                const Index bra = (i % 4 < 2) ? 0 : 27;
                states[i][ket * active_solver.layout.hilbert_dim + bra] = Complex{1.0, 0.0};
            }

            return states;
        };

    for (Index test_batch_size : std::vector<Index>{1, 4, 8, 16, 32, 64}) {
        std::vector<std::vector<Complex>> timing_states =
            make_batch_initial_states(solver, test_batch_size);

        BatchEvolutionConfig timing_config{
            0.0,
            1.0e-3,
            1,
            test_batch_size
        };

        BatchEvolutionTiming cpu_timing =
            time_density_batch_cpu_reference(
                solver,
                timing_states,
                timing_config);

        BatchEvolutionTiming cuda_timing =
            time_density_batch_cuda_ts(
                solver,
                timing_states,
                timing_config);

        std::cout << "Batch size " << test_batch_size
                  << " CPU time (s): " << cpu_timing.elapsed_seconds << std::endl;

        std::cout << "Batch size " << test_batch_size
                  << " CUDA wall time (s): " << cuda_timing.elapsed_seconds << std::endl;

        const double cpu_states_per_second =
            static_cast<double>(test_batch_size) / cpu_timing.elapsed_seconds;

        const double cuda_states_per_second =
            static_cast<double>(test_batch_size) / cuda_timing.elapsed_seconds;

        std::cout << "Batch size " << test_batch_size
                  << " CPU states/s: " << cpu_states_per_second << std::endl;

        std::cout << "Batch size " << test_batch_size
                  << " CUDA states/s: " << cuda_states_per_second << std::endl;

        std::cout << "Batch size " << test_batch_size
                  << " CUDA event time (s): " << cuda_timing.gpu_elapsed_seconds << std::endl;

        const double cuda_gpu_states_per_second =
            static_cast<double>(test_batch_size) / cuda_timing.gpu_elapsed_seconds;

        std::cout << "Batch size " << test_batch_size
                  << " CUDA event states/s: " << cuda_gpu_states_per_second << std::endl;

        std::cout << "Batch size " << test_batch_size
                  << " CUDA sample entry state 1 (27,0): "
                  << cuda_timing.result.final_states[std::min<Index>(1, test_batch_size - 1)]
                         .at(27 * solver.layout.hilbert_dim + 0)
                  << std::endl;
    }

    std::cout << "\n===== Batched grouped-layout smoke test =====\n" << std::endl;

    GroupedStateLayout single_grouped_layout =
        make_grouped_state_layout(target_sites, local_dims);

    BatchedGroupedLayout batched_layout =
        make_batched_grouped_layout(single_grouped_layout, 4);

    std::vector<std::vector<Complex>> flat_batch_states(
        4,
        std::vector<Complex>(solver.layout.density_dim, Complex{0.0, 0.0}));

    flat_batch_states[0][0 * solver.layout.hilbert_dim + 0] = Complex{1.0, 0.0};
    flat_batch_states[1][0 * solver.layout.hilbert_dim + 27] = Complex{1.0, 0.0};
    flat_batch_states[2][27 * solver.layout.hilbert_dim + 0] = Complex{1.0, 0.0};
    flat_batch_states[3][27 * solver.layout.hilbert_dim + 27] = Complex{1.0, 0.0};

    std::vector<Complex> grouped_batch;
    pack_flat_batch_to_grouped_batch(
        batched_layout,
        flat_batch_states,
        grouped_batch);

    std::vector<std::vector<Complex>> flat_batch_roundtrip;
    unpack_grouped_batch_to_flat_batch(
        batched_layout,
        grouped_batch,
        flat_batch_roundtrip);

    std::cout << "Batched grouped roundtrip entry state 1 (0,27): "
              << flat_batch_roundtrip[1].at(0 * solver.layout.hilbert_dim + 27)
              << std::endl;

    std::cout << "\n===== Batched grouped-left smoke test =====\n" << std::endl;

    std::vector<std::vector<Complex>> batched_left_out =
        apply_batched_grouped_left_cuda_prototype(
            solver,
            zzz_three_site,
            target_sites,
            flat_batch_states);

    std::vector<std::vector<Complex>> batched_left_out_device_staged =
        apply_batched_grouped_left_cuda_device_staged(
            solver,
            zzz_three_site,
            target_sites,
            flat_batch_states);

    std::vector<std::vector<Complex>> batched_left_out_fused =
        apply_batched_grouped_left_cuda_fused_candidate(
            solver,
            zzz_three_site,
            target_sites,
            flat_batch_states);

    std::cout << "Batched grouped-left state 1 entry (0,27): "
              << batched_left_out[1].at(0 * solver.layout.hilbert_dim + 27)
              << std::endl;

    std::cout << "Batched device-staged grouped-left state 1 entry (0,27): "
              << batched_left_out_device_staged[1].at(0 * solver.layout.hilbert_dim + 27)
              << std::endl;

    std::cout << "Batched fused grouped-left state 1 entry (0,27): "
              << batched_left_out_fused[1].at(0 * solver.layout.hilbert_dim + 27)
              << std::endl;

    std::cout << "\n===== Batched grouped-left timing comparison =====\n" << std::endl;

    for (Index grouped_left_batch_size : std::vector<Index>{4, 16, 32, 64, 128, 256}) {
        std::vector<std::vector<Complex>> grouped_left_states(
            grouped_left_batch_size,
            std::vector<Complex>(solver.layout.density_dim, Complex{0.0, 0.0}));

        for (Index i = 0; i < grouped_left_batch_size; ++i) {
            const Index ket = (i % 2 == 0) ? 0 : 27;
            const Index bra = (i % 4 < 2) ? 0 : 27;
            grouped_left_states[i][ket * solver.layout.hilbert_dim + bra] = Complex{1.0, 0.0};
        }

        BatchedGroupedApplyTiming prototype_timing =
            time_batched_grouped_left_cuda_prototype(
                solver,
                zzz_three_site,
                target_sites,
                grouped_left_states);

        BatchedGroupedApplyTiming device_staged_timing =
            time_batched_grouped_left_cuda_device_staged(
                solver,
                zzz_three_site,
                target_sites,
                grouped_left_states);

        BatchedGroupedApplyTiming batch_object_timing =
            time_batched_grouped_left_cuda_batch_object(
                solver,
                zzz_three_site,
                target_sites,
                grouped_left_states);

        BatchedGroupedApplyTiming fused_candidate_timing =
            time_batched_grouped_left_cuda_fused_candidate(
                solver,
                zzz_three_site,
                target_sites,
                grouped_left_states);

        const double prototype_states_per_second =
            static_cast<double>(grouped_left_batch_size) / prototype_timing.wall_seconds;
        const double device_staged_states_per_second =
            static_cast<double>(grouped_left_batch_size) / device_staged_timing.wall_seconds;
        const double batch_object_states_per_second =
            static_cast<double>(grouped_left_batch_size) / batch_object_timing.wall_seconds;
        const double fused_candidate_states_per_second =
            static_cast<double>(grouped_left_batch_size) / fused_candidate_timing.wall_seconds;

        std::cout << "Grouped-left batch size " << grouped_left_batch_size
                  << " prototype wall time (s): "
                  << prototype_timing.wall_seconds << std::endl;

        std::cout << "Grouped-left batch size " << grouped_left_batch_size
                  << " prototype event time (s): "
                  << prototype_timing.gpu_seconds << std::endl;

        std::cout << "Grouped-left batch size " << grouped_left_batch_size
                  << " device-staged wall time (s): "
                  << device_staged_timing.wall_seconds << std::endl;

        std::cout << "Grouped-left batch size " << grouped_left_batch_size
                  << " device-staged event time (s): "
                  << device_staged_timing.gpu_seconds << std::endl;

        std::cout << "Grouped-left batch size " << grouped_left_batch_size
                  << " prototype states/s: "
                  << prototype_states_per_second << std::endl;

        std::cout << "Grouped-left batch size " << grouped_left_batch_size
                  << " device-staged states/s: "
                  << device_staged_states_per_second << std::endl;

        std::cout << "Grouped-left batch size " << grouped_left_batch_size
                  << " device-staged sample entry state 1 (27,0): "
                  << device_staged_timing.output_states[std::min<Index>(1, grouped_left_batch_size - 1)]
                         .at(27 * solver.layout.hilbert_dim + 0)
                  << std::endl;

        const double device_staged_speedup =
            prototype_timing.wall_seconds / device_staged_timing.wall_seconds;

        std::cout << "Grouped-left batch size " << grouped_left_batch_size
                  << " device-staged speedup vs prototype: "
                  << device_staged_speedup << std::endl;

        std::cout << "Grouped-left batch size " << grouped_left_batch_size
                  << " batch-object wall time (s): "
                  << batch_object_timing.wall_seconds << std::endl;

        std::cout << "Grouped-left batch size " << grouped_left_batch_size
                  << " batch-object event time (s): "
                  << batch_object_timing.gpu_seconds << std::endl;

        std::cout << "Grouped-left batch size " << grouped_left_batch_size
                  << " batch-object states/s: "
                  << batch_object_states_per_second << std::endl;

        std::cout << "Grouped-left batch size " << grouped_left_batch_size
                  << " batch-object sample entry state 1 (27,0): "
                  << batch_object_timing.output_states[std::min<Index>(1, grouped_left_batch_size - 1)]
                         .at(27 * solver.layout.hilbert_dim + 0)
                  << std::endl;

        std::cout << "Grouped-left batch size " << grouped_left_batch_size
                  << " fused-candidate wall time (s): "
                  << fused_candidate_timing.wall_seconds << std::endl;

        std::cout << "Grouped-left batch size " << grouped_left_batch_size
                  << " fused-candidate event time (s): "
                  << fused_candidate_timing.gpu_seconds << std::endl;

        std::cout << "Grouped-left batch size " << grouped_left_batch_size
                  << " fused-candidate states/s: "
                  << fused_candidate_states_per_second << std::endl;

        std::cout << "Grouped-left batch size " << grouped_left_batch_size
                  << " fused-candidate sample entry state 1 (27,0): "
                  << fused_candidate_timing.output_states[std::min<Index>(1, grouped_left_batch_size - 1)]
                         .at(27 * solver.layout.hilbert_dim + 0)
                  << std::endl;
    }

    std::cout << "\n===== Grouped-left batch timing summary =====\n" << std::endl;
    std::cout << "Inspect prototype vs device-staged vs fused-candidate trends." << std::endl;

    auto run_large_grouped_left_benchmark =
        [&](const std::vector<Index>& large_local_dims,
            const std::vector<Index>& batch_sizes,
            const char* label) {
            std::cout << "\n===== Larger-system grouped-left benchmark: "
                      << label << " =====\n" << std::endl;

            std::vector<Complex> large_diag_op = make_zzz_three_site();

            OperatorTerm large_h_term{
                TermKind::Hamiltonian,
                "large_q123_zzz",
                {0, 1, 2},
                large_diag_op,
                27,
                27
            };

            Model large_model{
                large_local_dims,
                {large_h_term},
                {},
                {}
            };

            Solver large_solver = make_solver(large_model);
            const std::vector<Index> large_target_sites = {0, 1, 2};

            std::cout << "Large benchmark Hilbert dimension: "
                      << large_solver.layout.hilbert_dim << std::endl;

            std::cout << "Large benchmark density dimension: "
                      << large_solver.layout.density_dim << std::endl;

            for (Index large_batch_size : batch_sizes) {
                std::vector<std::vector<Complex>> large_states(
                    large_batch_size,
                    std::vector<Complex>(large_solver.layout.density_dim, Complex{0.0, 0.0}));

                for (Index i = 0; i < large_batch_size; ++i) {
                    const Index ket = (i % 2 == 0) ? 0 : 27;
                    const Index bra = (i % 4 < 2) ? 0 : 27;
                    large_states[i][ket * large_solver.layout.hilbert_dim + bra] = Complex{1.0, 0.0};
                }

                BatchedGroupedApplyTiming large_prototype_timing =
                    time_batched_grouped_left_cuda_prototype(
                        large_solver,
                        large_diag_op,
                        large_target_sites,
                        large_states);

                BatchedGroupedApplyTiming large_device_staged_timing =
                    time_batched_grouped_left_cuda_device_staged(
                        large_solver,
                        large_diag_op,
                        large_target_sites,
                        large_states);

                BatchedGroupedApplyTiming large_fused_timing =
                    time_batched_grouped_left_cuda_fused_candidate(
                        large_solver,
                        large_diag_op,
                        large_target_sites,
                        large_states);

                const double large_prototype_states_per_second =
                    static_cast<double>(large_batch_size) / large_prototype_timing.wall_seconds;
                const double large_device_staged_states_per_second =
                    static_cast<double>(large_batch_size) / large_device_staged_timing.wall_seconds;
                const double large_fused_states_per_second =
                    static_cast<double>(large_batch_size) / large_fused_timing.wall_seconds;

                std::cout << "Large grouped-left batch size " << large_batch_size
                          << " prototype wall time (s): "
                          << large_prototype_timing.wall_seconds << std::endl;

                std::cout << "Large grouped-left batch size " << large_batch_size
                          << " device-staged wall time (s): "
                          << large_device_staged_timing.wall_seconds << std::endl;

                std::cout << "Large grouped-left batch size " << large_batch_size
                          << " fused-candidate wall time (s): "
                          << large_fused_timing.wall_seconds << std::endl;

                std::cout << "Large grouped-left batch size " << large_batch_size
                          << " prototype states/s: "
                          << large_prototype_states_per_second << std::endl;

                std::cout << "Large grouped-left batch size " << large_batch_size
                          << " device-staged states/s: "
                          << large_device_staged_states_per_second << std::endl;

                std::cout << "Large grouped-left batch size " << large_batch_size
                          << " fused-candidate states/s: "
                          << large_fused_states_per_second << std::endl;

                std::cout << "Large grouped-left batch size " << large_batch_size
                          << " fused-candidate sample entry state 1 (27,0): "
                          << large_fused_timing.output_states[std::min<Index>(1, large_batch_size - 1)]
                                 .at(27 * large_solver.layout.hilbert_dim + 0)
                          << std::endl;
            }
        };

    run_large_grouped_left_benchmark(
        {3, 3, 3, 3, 3},
        {16, 64, 128},
        "n=5");

    run_large_grouped_left_benchmark(
        {3, 3, 3, 3, 3, 3},
        {16, 64},
        "n=6");

    PetscFinalize();
    return 0;
}
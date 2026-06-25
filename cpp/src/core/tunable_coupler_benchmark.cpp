#include "culindblad/tunable_coupler_benchmark.hpp"

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <petscts.h>
#include <petscvec.h>

#include "culindblad/backend.hpp"
#include "culindblad/batch_evolution.hpp"
#include "culindblad/petsc_cuda_ts_rhs.hpp"
#include "culindblad/state_buffer.hpp"
#include "culindblad/time_dependence.hpp"
#include "culindblad/time_dependent_term.hpp"

namespace culindblad {

namespace {

// ---------------------------------------------------------------------------
// Local single-site operators (Fock basis, cutoff d).
// ---------------------------------------------------------------------------

std::vector<Complex> make_local_annihilation_operator(Index d)
{
    std::vector<Complex> a(d * d, Complex{0.0, 0.0});
    for (Index n = 1; n < d; ++n) {
        a[(n - 1) * d + n] = Complex{std::sqrt(static_cast<double>(n)), 0.0};
    }
    return a;
}

std::vector<Complex> make_local_number_operator(Index d)
{
    std::vector<Complex> n_op(d * d, Complex{0.0, 0.0});
    for (Index n = 0; n < d; ++n) {
        n_op[n * d + n] = Complex{static_cast<double>(n), 0.0};
    }
    return n_op;
}

// 0.5 * n (n - 1) = 0.5 * b^dag b^dag b b  (the anharmonicity operator, /alpha).
std::vector<Complex> make_local_number_squared_shifted_operator(Index d)
{
    std::vector<Complex> op(d * d, Complex{0.0, 0.0});
    for (Index n = 0; n < d; ++n) {
        const double value =
            0.5 * static_cast<double>(n) * static_cast<double>(n - 1);
        op[n * d + n] = Complex{value, 0.0};
    }
    return op;
}

// Two-site FULL capacitive coupling operator on the d*d local product space,
// flatten(n0, n1) = n0*d + n1 (site[0] = major). Builds (without the scalar g):
//   a0^dag a1 + a0 a1^dag - a0^dag a1^dag - a0 a1
// This is Hermitian and symmetric under site swap, so coupling on {i, j} is
// orientation-independent.
std::vector<Complex> make_two_site_full_coupling_operator(Index d)
{
    const Index dd = d * d;
    std::vector<Complex> op(dd * dd, Complex{0.0, 0.0});

    auto flatten = [d](Index left, Index right) {
        return left * d + right;
    };

    for (Index n0 = 0; n0 < d; ++n0) {
        for (Index n1 = 0; n1 < d; ++n1) {
            const Index col = flatten(n0, n1);

            // + a0^dag a1 : n0 -> n0+1, n1 -> n1-1
            if (n0 + 1 < d && n1 > 0) {
                const Index row = flatten(n0 + 1, n1 - 1);
                const double coeff =
                    std::sqrt(static_cast<double>(n0 + 1)) *
                    std::sqrt(static_cast<double>(n1));
                op[row * dd + col] += Complex{coeff, 0.0};
            }

            // + a0 a1^dag : n0 -> n0-1, n1 -> n1+1
            if (n0 > 0 && n1 + 1 < d) {
                const Index row = flatten(n0 - 1, n1 + 1);
                const double coeff =
                    std::sqrt(static_cast<double>(n0)) *
                    std::sqrt(static_cast<double>(n1 + 1));
                op[row * dd + col] += Complex{coeff, 0.0};
            }

            // - a0^dag a1^dag : n0 -> n0+1, n1 -> n1+1
            if (n0 + 1 < d && n1 + 1 < d) {
                const Index row = flatten(n0 + 1, n1 + 1);
                const double coeff =
                    std::sqrt(static_cast<double>(n0 + 1)) *
                    std::sqrt(static_cast<double>(n1 + 1));
                op[row * dd + col] -= Complex{coeff, 0.0};
            }

            // - a0 a1 : n0 -> n0-1, n1 -> n1-1
            if (n0 > 0 && n1 > 0) {
                const Index row = flatten(n0 - 1, n1 - 1);
                const double coeff =
                    std::sqrt(static_cast<double>(n0)) *
                    std::sqrt(static_cast<double>(n1));
                op[row * dd + col] -= Complex{coeff, 0.0};
            }
        }
    }

    return op;
}

// Rotating-frame decomposition of the two-site full capacitive coupling.
// In the frame H_R = sum omega_i n_i, the operator op = (a0^dag a1 + a0 a1^dag)
//   - (a0^dag a1^dag + a0 a1) rotates to
//   cos(D t) H1 + sin(D t) H2 - cos(S t) H3 - sin(S t) H4,
// where D = omega0 - omega1 (detuning), S = omega0 + omega1 (sum), and
//   H1 = a0^dag a1 + a0 a1^dag                 (co-rotating, real symmetric)
//   H2 = i (a0^dag a1 - a0 a1^dag)             (co-rotating, Hermitian)
//   H3 = a0^dag a1^dag + a0 a1                 (counter-rotating, real symmetric)
//   H4 = i (a0^dag a1^dag - a0 a1)             (counter-rotating, Hermitian)
// Each is a Hermitian d*d x d*d operator; the four are applied with real, time-
// dependent coefficients so that op_rotated(0) = H1 - H3 = op (matches lab frame).
struct TwoSiteRotatedCoupling {
    std::vector<Complex> h1;  // a0^dag a1 + a0 a1^dag
    std::vector<Complex> h2;  // i (a0^dag a1 - a0 a1^dag)
    std::vector<Complex> h3;  // a0^dag a1^dag + a0 a1
    std::vector<Complex> h4;  // i (a0^dag a1^dag - a0 a1)
};

TwoSiteRotatedCoupling make_two_site_rotated_coupling_operators(Index d)
{
    const Index dd = d * d;
    TwoSiteRotatedCoupling r;
    r.h1.assign(dd * dd, Complex{0.0, 0.0});
    r.h2.assign(dd * dd, Complex{0.0, 0.0});
    r.h3.assign(dd * dd, Complex{0.0, 0.0});
    r.h4.assign(dd * dd, Complex{0.0, 0.0});

    auto flatten = [d](Index left, Index right) { return left * d + right; };

    for (Index n0 = 0; n0 < d; ++n0) {
        for (Index n1 = 0; n1 < d; ++n1) {
            const Index col = flatten(n0, n1);

            // M1 = a0^dag a1 : n0->n0+1, n1->n1-1
            if (n0 + 1 < d && n1 > 0) {
                const Index row = flatten(n0 + 1, n1 - 1);
                const double c = std::sqrt(static_cast<double>(n0 + 1)) *
                                 std::sqrt(static_cast<double>(n1));
                r.h1[row * dd + col] += Complex{c, 0.0};   // H1 += M1
                r.h2[row * dd + col] += Complex{0.0, c};   // H2 += i*M1
            }
            // M2 = a0 a1^dag : n0->n0-1, n1->n1+1  (= M1^dag)
            if (n0 > 0 && n1 + 1 < d) {
                const Index row = flatten(n0 - 1, n1 + 1);
                const double c = std::sqrt(static_cast<double>(n0)) *
                                 std::sqrt(static_cast<double>(n1 + 1));
                r.h1[row * dd + col] += Complex{c, 0.0};   // H1 += M2
                r.h2[row * dd + col] += Complex{0.0, -c};  // H2 += -i*M2
            }
            // M3 = a0^dag a1^dag : n0->n0+1, n1->n1+1
            if (n0 + 1 < d && n1 + 1 < d) {
                const Index row = flatten(n0 + 1, n1 + 1);
                const double c = std::sqrt(static_cast<double>(n0 + 1)) *
                                 std::sqrt(static_cast<double>(n1 + 1));
                r.h3[row * dd + col] += Complex{c, 0.0};   // H3 += M3
                r.h4[row * dd + col] += Complex{0.0, c};   // H4 += i*M3
            }
            // M4 = a0 a1 : n0->n0-1, n1->n1-1  (= M3^dag)
            if (n0 > 0 && n1 > 0) {
                const Index row = flatten(n0 - 1, n1 - 1);
                const double c = std::sqrt(static_cast<double>(n0)) *
                                 std::sqrt(static_cast<double>(n1));
                r.h3[row * dd + col] += Complex{c, 0.0};   // H3 += M4
                r.h4[row * dd + col] += Complex{0.0, -c};  // H4 += -i*M4
            }
        }
    }
    return r;
}

// ---------------------------------------------------------------------------
// CPU execution (fixed-step TSRK4, mirrors the validated chain benchmark path).
// ---------------------------------------------------------------------------

struct CpuTsRhsContext {
    const Solver* solver = nullptr;
};

PetscErrorCode ts_rhs_function_cpu_full_model_liouvillian(
    TS,
    PetscReal t,
    Vec x,
    Vec f,
    void* ctx)
{
    auto* rhs_ctx = static_cast<CpuTsRhsContext*>(ctx);
    if (!rhs_ctx || !rhs_ctx->solver) {
        return PETSC_ERR_ARG_NULL;
    }

    const Solver& solver = *rhs_ctx->solver;

    const PetscScalar* x_ptr = nullptr;
    PetscScalar* f_ptr = nullptr;

    PetscCall(VecGetArrayRead(x, &x_ptr));
    PetscCall(VecGetArray(f, &f_ptr));

    ConstStateBuffer in_buf{
        reinterpret_cast<const Complex*>(x_ptr),
        solver.layout.density_dim};
    StateBuffer out_buf{
        reinterpret_cast<Complex*>(f_ptr),
        solver.layout.density_dim};

    apply_liouvillian_at_time(solver, static_cast<double>(t), in_buf, out_buf);

    PetscCall(VecRestoreArray(f, &f_ptr));
    PetscCall(VecRestoreArrayRead(x, &x_ptr));
    return 0;
}

struct CpuBenchmarkExecutionContext {
    TS ts = nullptr;
    Vec x = nullptr;
    CpuTsRhsContext rhs_ctx{};
    bool initialized = false;
};

PetscErrorCode create_cpu_benchmark_execution_context(
    const Solver& solver,
    CpuBenchmarkExecutionContext& ctx)
{
    ctx.ts = nullptr;
    ctx.x = nullptr;
    ctx.initialized = false;
    ctx.rhs_ctx.solver = &solver;

    PetscCall(TSCreate(PETSC_COMM_SELF, &ctx.ts));
    PetscCall(TSSetType(ctx.ts, TSRK));
    PetscCall(TSRKSetType(ctx.ts, TSRK4));
    TSAdapt cpu_adapt = nullptr;
    PetscCall(TSGetAdapt(ctx.ts, &cpu_adapt));
    PetscCall(TSAdaptSetType(cpu_adapt, TSADAPTNONE));

    PetscCall(VecCreate(PETSC_COMM_SELF, &ctx.x));
    PetscCall(VecSetSizes(ctx.x, PETSC_DECIDE, solver.layout.density_dim));
    PetscCall(VecSetFromOptions(ctx.x));
    PetscCall(VecSet(ctx.x, 0.0));

    PetscCall(TSSetRHSFunction(
        ctx.ts, nullptr, ts_rhs_function_cpu_full_model_liouvillian, &ctx.rhs_ctx));

    PetscCall(TSSetMaxSteps(ctx.ts, 1000000));
    PetscCall(TSSetExactFinalTime(ctx.ts, TS_EXACTFINALTIME_MATCHSTEP));
    PetscCall(TSSetFromOptions(ctx.ts));

    ctx.initialized = true;
    return 0;
}

PetscErrorCode destroy_cpu_benchmark_execution_context(
    CpuBenchmarkExecutionContext& ctx)
{
    if (!ctx.initialized) {
        return 0;
    }
    if (ctx.x != nullptr) {
        PetscCall(VecDestroy(&ctx.x));
    }
    if (ctx.ts != nullptr) {
        PetscCall(TSDestroy(&ctx.ts));
    }
    ctx.initialized = false;
    return 0;
}

std::vector<Complex> evolve_single_state_cpu_ts(
    const Solver& solver,
    const std::vector<Complex>& initial_state,
    double t0,
    double tfinal,
    Index num_steps,
    CpuBenchmarkExecutionContext& ctx)
{
    if (!ctx.initialized) {
        throw std::runtime_error(
            "evolve_single_state_cpu_ts: CPU benchmark context not initialized");
    }
    if (num_steps == 0) {
        throw std::invalid_argument(
            "evolve_single_state_cpu_ts: num_steps must be > 0");
    }

    PetscCallAbort(PETSC_COMM_SELF, VecSet(ctx.x, 0.0));

    PetscScalar* x_ptr = nullptr;
    PetscCallAbort(PETSC_COMM_SELF, VecGetArray(ctx.x, &x_ptr));
    for (Index i = 0; i < solver.layout.density_dim; ++i) {
        x_ptr[i] = initial_state[i];
    }
    PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(ctx.x, &x_ptr));

    const double dt_initial = (tfinal - t0) / static_cast<double>(num_steps);

    PetscCallAbort(PETSC_COMM_SELF, TSSetTime(ctx.ts, t0));
    PetscCallAbort(PETSC_COMM_SELF, TSSetStepNumber(ctx.ts, 0));
    PetscCallAbort(PETSC_COMM_SELF, TSSetTimeStep(ctx.ts, dt_initial));
    PetscCallAbort(PETSC_COMM_SELF, TSSetMaxSteps(ctx.ts, static_cast<PetscInt>(num_steps)));
    PetscCallAbort(PETSC_COMM_SELF, TSSetMaxTime(ctx.ts, tfinal));
    PetscCallAbort(PETSC_COMM_SELF, TSSolve(ctx.ts, ctx.x));

    PetscCallAbort(PETSC_COMM_SELF, VecGetArray(ctx.x, &x_ptr));
    std::vector<Complex> final_state(solver.layout.density_dim, Complex{0.0, 0.0});
    for (Index i = 0; i < solver.layout.density_dim; ++i) {
        final_state[i] = x_ptr[i];
    }
    PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(ctx.x, &x_ptr));

    return final_state;
}

// ---------------------------------------------------------------------------
// CUDA batched execution (fixed-step TSRK4, mirrors the validated chain path).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Custom fixed-step integrators that bypass PETSc TS and drive the matrix-free
// Liouvillian L(t) directly. Selected via -tc_integrator: 0 = PETSc TSRK4
// (default), 1 = custom classic RK4 (3 extra registers, bit-identical to TSRK4),
// 2 = Taylor exp-midpoint (exponential integrator, exp(dt L(t+dt/2)) per step).
// The custom path is the stepping foundation for the multi-GPU work and lets us
// benchmark memory (register count) + the exponential integrator head-to-head.
// ---------------------------------------------------------------------------
enum class CustomIntegrator { TsRk4 = 0, Rk4Custom = 1, ExpmvMidpoint = 2 };

using TcRhsFn = PetscErrorCode (*)(TS, PetscReal, Vec, Vec, void*);

// f = RHS(t) x via the chosen matrix-free RHS (TS arg unused -> nullptr). Defaults to the full
// Liouvillian; the Schrodinger solver passes ts_rhs_function_cuda_full_model_schrodinger so the
// same low-overhead custom stepper drives |psi> (where PETSc TS bookkeeping is a bigger fraction
// of the per-step cost than for the large density state).
inline void apply_rhs(
    CudaBatchExecutionContext& ctx, double t, Vec in, Vec out, long& matvecs, TcRhsFn rhs_fn)
{
    PetscCallAbort(
        PETSC_COMM_SELF,
        rhs_fn(nullptr, static_cast<PetscReal>(t), in, out, &ctx.rhs_ctx));
    ++matvecs;
}

void evolve_batch_custom(
    CudaBatchExecutionContext& ctx,
    double t0,
    double tfinal,
    Index num_steps,
    CustomIntegrator kind,
    int expmv_order,
    TcRhsFn rhs_fn = ts_rhs_function_cuda_full_model_liouvillian)
{
    Vec y = ctx.x;  // evolved in place
    Vec a = nullptr;
    Vec b = nullptr;
    Vec c = nullptr;
    PetscCallAbort(PETSC_COMM_SELF, VecDuplicate(y, &a));
    PetscCallAbort(PETSC_COMM_SELF, VecDuplicate(y, &b));
    PetscCallAbort(PETSC_COMM_SELF, VecDuplicate(y, &c));

    const double dt = (tfinal - t0) / static_cast<double>(num_steps);
    long matvecs = 0;
    const auto wall0 = std::chrono::steady_clock::now();

    for (Index step = 0; step < num_steps; ++step) {
        const double t = t0 + static_cast<double>(step) * dt;
        if (kind == CustomIntegrator::Rk4Custom) {
            // Classic RK4: a = weighted accumulator, b = stage input, c = stage k.
            apply_rhs(ctx, t, y, c, matvecs, rhs_fn);
            PetscCallAbort(PETSC_COMM_SELF, VecCopy(c, a));                 // a = k1
            PetscCallAbort(PETSC_COMM_SELF, VecWAXPY(b, dt * 0.5, c, y));   // b = y + dt/2 k1
            apply_rhs(ctx, t + dt * 0.5, b, c, matvecs, rhs_fn);
            PetscCallAbort(PETSC_COMM_SELF, VecAXPY(a, 2.0, c));            // a += 2 k2
            PetscCallAbort(PETSC_COMM_SELF, VecWAXPY(b, dt * 0.5, c, y));   // b = y + dt/2 k2
            apply_rhs(ctx, t + dt * 0.5, b, c, matvecs, rhs_fn);
            PetscCallAbort(PETSC_COMM_SELF, VecAXPY(a, 2.0, c));            // a += 2 k3
            PetscCallAbort(PETSC_COMM_SELF, VecWAXPY(b, dt, c, y));         // b = y + dt k3
            apply_rhs(ctx, t + dt, b, c, matvecs, rhs_fn);
            PetscCallAbort(PETSC_COMM_SELF, VecAXPY(a, 1.0, c));            // a += k4
            PetscCallAbort(PETSC_COMM_SELF, VecAXPY(y, dt / 6.0, a));       // y += dt/6 a
        } else {
            // Taylor exp-midpoint: y <- exp(dt L(t+dt/2)) y, L frozen at the midpoint.
            // a = running result, b = current Taylor term, c = L applied to b.
            const double tm = t + dt * 0.5;
            PetscCallAbort(PETSC_COMM_SELF, VecCopy(y, a));  // result = y
            PetscCallAbort(PETSC_COMM_SELF, VecCopy(y, b));  // term   = y
            for (int k = 1; k <= expmv_order; ++k) {
                apply_rhs(ctx, tm, b, c, matvecs, rhs_fn);            // c = L term
                PetscCallAbort(PETSC_COMM_SELF, VecCopy(c, b));
                PetscCallAbort(
                    PETSC_COMM_SELF, VecScale(b, dt / static_cast<double>(k)));  // term = dt/k L term
                PetscCallAbort(PETSC_COMM_SELF, VecAXPY(a, 1.0, b));  // result += term
                PetscReal term_norm = 0.0;
                PetscReal res_norm = 0.0;
                PetscCallAbort(PETSC_COMM_SELF, VecNorm(b, NORM_2, &term_norm));
                PetscCallAbort(PETSC_COMM_SELF, VecNorm(a, NORM_2, &res_norm));
                if (term_norm <= 1.0e-15 * res_norm) {
                    break;
                }
            }
            PetscCallAbort(PETSC_COMM_SELF, VecCopy(a, y));  // y = result
        }
    }

    const auto wall1 = std::chrono::steady_clock::now();
    const double wall_ms =
        std::chrono::duration<double, std::milli>(wall1 - wall0).count();
    std::fprintf(
        stderr,
        "INTEGRATOR_BENCH kind=%d steps=%ld matvecs=%ld matvecs_per_step=%.2f "
        "wall_ms=%.1f extra_registers=3\n",
        static_cast<int>(kind), static_cast<long>(num_steps), matvecs,
        static_cast<double>(matvecs) / static_cast<double>(num_steps), wall_ms);

    PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&a));
    PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&b));
    PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&c));
}

std::vector<std::vector<Complex>> evolve_state_batch_cuda_ts(
    const Solver& solver,
    const std::vector<std::vector<Complex>>& initial_states,
    double t0,
    double tfinal,
    Index num_steps,
    CudaBatchExecutionContext& ctx)
{
    if (!ctx.initialized || ctx.ts == nullptr || ctx.x == nullptr) {
        throw std::runtime_error(
            "evolve_state_batch_cuda_ts: CUDA benchmark context not initialized");
    }
    if (num_steps == 0) {
        throw std::invalid_argument(
            "evolve_state_batch_cuda_ts: num_steps must be > 0");
    }
    if (initial_states.size() != ctx.rhs_ctx.batch_size) {
        throw std::invalid_argument(
            "evolve_state_batch_cuda_ts: initial state count must match context batch size");
    }

    PetscCallAbort(PETSC_COMM_SELF, VecSet(ctx.x, 0.0));

    PetscScalar* x_ptr = nullptr;
    PetscCallAbort(PETSC_COMM_SELF, VecGetArray(ctx.x, &x_ptr));
    for (Index batch = 0; batch < initial_states.size(); ++batch) {
        if (initial_states[batch].size() != solver.layout.density_dim) {
            PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(ctx.x, &x_ptr));
            throw std::invalid_argument(
                "evolve_state_batch_cuda_ts: initial state size mismatch");
        }
        for (Index i = 0; i < solver.layout.density_dim; ++i) {
            x_ptr[batch * solver.layout.density_dim + i] = initial_states[batch][i];
        }
    }
    PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(ctx.x, &x_ptr));

    const double dt_initial = (tfinal - t0) / static_cast<double>(num_steps);

    PetscInt integrator = 0;
    PetscCallAbort(PETSC_COMM_SELF, PetscOptionsGetInt(
        nullptr, nullptr, "-tc_integrator", &integrator, nullptr));
    if (integrator == 0) {
        PetscCallAbort(PETSC_COMM_SELF, TSSetTime(ctx.ts, t0));
        PetscCallAbort(PETSC_COMM_SELF, TSSetStepNumber(ctx.ts, 0));
        PetscCallAbort(PETSC_COMM_SELF, TSSetTimeStep(ctx.ts, dt_initial));
        PetscCallAbort(PETSC_COMM_SELF, TSSetMaxSteps(ctx.ts, static_cast<PetscInt>(num_steps)));
        PetscCallAbort(PETSC_COMM_SELF, TSSetMaxTime(ctx.ts, tfinal));
        PetscCallAbort(PETSC_COMM_SELF, TSSolve(ctx.ts, ctx.x));
    } else {
        PetscInt expmv_order = 24;
        PetscCallAbort(PETSC_COMM_SELF, PetscOptionsGetInt(
            nullptr, nullptr, "-tc_expmv_order", &expmv_order, nullptr));
        evolve_batch_custom(
            ctx, t0, tfinal, num_steps,
            (integrator == 1) ? CustomIntegrator::Rk4Custom
                              : CustomIntegrator::ExpmvMidpoint,
            static_cast<int>(expmv_order));
    }

    PetscCallAbort(PETSC_COMM_SELF, VecGetArray(ctx.x, &x_ptr));
    std::vector<std::vector<Complex>> final_states(
        initial_states.size(),
        std::vector<Complex>(solver.layout.density_dim, Complex{0.0, 0.0}));
    for (Index batch = 0; batch < initial_states.size(); ++batch) {
        for (Index i = 0; i < solver.layout.density_dim; ++i) {
            final_states[batch][i] = x_ptr[batch * solver.layout.density_dim + i];
        }
    }
    PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(ctx.x, &x_ptr));

    return final_states;
}

std::vector<std::vector<Complex>> make_selected_basis_density_states(
    const Solver& solver,
    const std::vector<Index>& selected_state_indices)
{
    std::vector<std::vector<Complex>> states;
    states.reserve(selected_state_indices.size());
    for (Index state_index : selected_state_indices) {
        if (state_index >= solver.layout.density_dim) {
            throw std::invalid_argument(
                "make_selected_basis_density_states: selected state index out of range");
        }
        std::vector<Complex> rho(solver.layout.density_dim, Complex{0.0, 0.0});
        rho[state_index] = Complex{1.0, 0.0};
        states.push_back(std::move(rho));
    }
    return states;
}

// State-vector evolve via PETSc TSRK4 on the registered Schrodinger RHS (memory is a non-issue
// for |psi>, so the custom low-storage integrator is unnecessary). Mirrors evolve_state_batch_
// cuda_ts but with hilbert_dim (D) stride and TSRK4 set internally.
std::vector<std::vector<Complex>> evolve_state_vector_batch_cuda_ts(
    const Solver& solver,
    const std::vector<std::vector<Complex>>& initial_states,
    double t0,
    double tfinal,
    Index num_steps,
    CudaBatchExecutionContext& ctx)
{
    if (!ctx.initialized || ctx.ts == nullptr || ctx.x == nullptr) {
        throw std::runtime_error(
            "evolve_state_vector_batch_cuda_ts: context not initialized");
    }
    if (num_steps == 0) {
        throw std::invalid_argument(
            "evolve_state_vector_batch_cuda_ts: num_steps must be > 0");
    }
    if (initial_states.size() != ctx.rhs_ctx.batch_size) {
        throw std::invalid_argument(
            "evolve_state_vector_batch_cuda_ts: initial state count must match batch size");
    }
    const Index hd = solver.layout.hilbert_dim;

    PetscCallAbort(PETSC_COMM_SELF, VecSet(ctx.x, 0.0));
    PetscScalar* x_ptr = nullptr;
    PetscCallAbort(PETSC_COMM_SELF, VecGetArray(ctx.x, &x_ptr));
    for (Index batch = 0; batch < initial_states.size(); ++batch) {
        if (initial_states[batch].size() != hd) {
            PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(ctx.x, &x_ptr));
            throw std::invalid_argument(
                "evolve_state_vector_batch_cuda_ts: initial state size mismatch");
        }
        for (Index i = 0; i < hd; ++i) {
            x_ptr[batch * hd + i] = initial_states[batch][i];
        }
    }
    PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(ctx.x, &x_ptr));

    // -tc_integrator: 0 = PETSc TSRK4, 1 = custom low-overhead classic RK4 driving the
    // Schrodinger RHS. The custom stepper removes the PETSc TS per-step bookkeeping, which is a
    // larger fraction of the cost for the tiny |psi> state than for the density state. Both are
    // fixed-step classic RK4 and agree to rounding level.
    PetscInt integrator = 0;
    PetscCallAbort(PETSC_COMM_SELF, PetscOptionsGetInt(
        nullptr, nullptr, "-tc_integrator", &integrator, nullptr));
    if (integrator == 0) {
        const double dt = (tfinal - t0) / static_cast<double>(num_steps);
        PetscCallAbort(PETSC_COMM_SELF, TSSetType(ctx.ts, TSRK));
        PetscCallAbort(PETSC_COMM_SELF, TSRKSetType(ctx.ts, TSRK4));
        TSAdapt adapt = nullptr;
        PetscCallAbort(PETSC_COMM_SELF, TSGetAdapt(ctx.ts, &adapt));
        PetscCallAbort(PETSC_COMM_SELF, TSAdaptSetType(adapt, TSADAPTNONE));
        PetscCallAbort(PETSC_COMM_SELF, TSSetTime(ctx.ts, t0));
        PetscCallAbort(PETSC_COMM_SELF, TSSetStepNumber(ctx.ts, 0));
        PetscCallAbort(PETSC_COMM_SELF, TSSetTimeStep(ctx.ts, dt));
        PetscCallAbort(PETSC_COMM_SELF, TSSetMaxSteps(ctx.ts, static_cast<PetscInt>(num_steps)));
        PetscCallAbort(PETSC_COMM_SELF, TSSetMaxTime(ctx.ts, tfinal));
        PetscCallAbort(PETSC_COMM_SELF, TSSetExactFinalTime(ctx.ts, TS_EXACTFINALTIME_MATCHSTEP));
        PetscCallAbort(PETSC_COMM_SELF, TSSolve(ctx.ts, ctx.x));
    } else {
        evolve_batch_custom(ctx, t0, tfinal, num_steps, CustomIntegrator::Rk4Custom, 0,
                            ts_rhs_function_cuda_full_model_schrodinger);
    }

    PetscCallAbort(PETSC_COMM_SELF, VecGetArray(ctx.x, &x_ptr));
    std::vector<std::vector<Complex>> final_states(
        initial_states.size(), std::vector<Complex>(hd, Complex{0.0, 0.0}));
    for (Index batch = 0; batch < initial_states.size(); ++batch) {
        for (Index i = 0; i < hd; ++i) {
            final_states[batch][i] = x_ptr[batch * hd + i];
        }
    }
    PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(ctx.x, &x_ptr));
    return final_states;
}

std::vector<std::vector<Complex>> make_selected_basis_state_vectors(
    const Solver& solver,
    const std::vector<Index>& selected_state_indices)
{
    std::vector<std::vector<Complex>> states;
    states.reserve(selected_state_indices.size());
    for (Index state_index : selected_state_indices) {
        if (state_index >= solver.layout.hilbert_dim) {
            throw std::invalid_argument(
                "make_selected_basis_state_vectors: selected state index out of range");
        }
        std::vector<Complex> psi(solver.layout.hilbert_dim, Complex{0.0, 0.0});
        psi[state_index] = Complex{1.0, 0.0};
        states.push_back(std::move(psi));
    }
    return states;
}

} // namespace

Index tunable_coupler_num_elements(const TunableCouplerBenchmarkConfig& config)
{
    return config.num_qubits + config.num_couplers;
}

Model build_tunable_coupler_model(const TunableCouplerBenchmarkConfig& config)
{
    const Index n_elem = tunable_coupler_num_elements(config);

    if (n_elem == 0) {
        throw std::invalid_argument(
            "build_tunable_coupler_model: need at least one element");
    }
    if (config.cutoff_dim < 2) {
        throw std::invalid_argument(
            "build_tunable_coupler_model: cutoff_dim must be >= 2");
    }
    if (config.omega.size() != n_elem ||
        config.alpha.size() != n_elem ||
        config.t1.size() != n_elem ||
        config.tphi.size() != n_elem) {
        throw std::invalid_argument(
            "build_tunable_coupler_model: per-element parameter size mismatch");
    }

    const Index d = config.cutoff_dim;
    std::vector<Index> local_dims(n_elem, d);

    const std::vector<Complex> a_local = make_local_annihilation_operator(d);
    const std::vector<Complex> n_local = make_local_number_operator(d);
    const std::vector<Complex> n2_shifted_local =
        make_local_number_squared_shifted_operator(d);
    const std::vector<Complex> coupling_local =
        make_two_site_full_coupling_operator(d);

    std::vector<OperatorTerm> hamiltonian_terms;
    std::vector<OperatorTerm> dissipator_terms;
    std::vector<TimeDependentTerm> time_dependent_terms;  // static model: empty

    // Drive helpers, shared by the rotating-frame couplings and the lab-frame drive
    // terms: the single-sided sine frequency modulation and the set of driven sites.
    constexpr double kPi = 3.14159265358979323846;
    const double drive_A = config.drive_amplitude;
    const double drive_T =
        (config.drive_t_gate > 0.0) ? config.drive_t_gate : config.tfinal;
    std::vector<bool> driven(n_elem, false);
    if (config.enable_drives) {
        for (Index s : config.driven_sites) {
            if (s >= n_elem) {
                throw std::invalid_argument(
                    "build_tunable_coupler_model: driven site out of range");
            }
            driven[s] = true;
        }
    }
    // omega_k(t) = omega_k - A*sin(pi t/T); delta_omega(t) is the deviation.
    auto delta_omega = [drive_A, drive_T, kPi](double t) -> double {
        if (t <= 0.0 || t >= drive_T) {
            return 0.0;
        }
        return -drive_A * std::sin(kPi * t / drive_T);
    };

    // Per-element bare Hamiltonian + dissipators.
    for (Index i = 0; i < n_elem; ++i) {
        // omega_i * n_i. In the rotating frame H_R = sum_i omega_i n_i this term is
        // exactly removed (the fast bare oscillation); only the slow anharmonicity and
        // drive modulation survive on the diagonal.
        if (!config.enable_rotating_frame) {
            std::vector<Complex> omega_term = n_local;
            for (Complex& x : omega_term) {
                x *= config.omega[i];
            }
            hamiltonian_terms.push_back(OperatorTerm{
                TermKind::Hamiltonian, "omega_n_" + std::to_string(i),
                {i}, omega_term, d, d});
        }

        std::vector<Complex> alpha_term = n2_shifted_local;
        for (Complex& x : alpha_term) {
            x *= config.alpha[i];
        }
        hamiltonian_terms.push_back(OperatorTerm{
            TermKind::Hamiltonian, "alpha_n2_" + std::to_string(i),
            {i}, alpha_term, d, d});

        std::vector<Complex> relax_op = a_local;
        for (Complex& x : relax_op) {
            x *= std::sqrt(1.0 / config.t1[i]);
        }
        dissipator_terms.push_back(OperatorTerm{
            TermKind::Dissipator, "relax_" + std::to_string(i),
            {i}, relax_op, d, d});

        std::vector<Complex> dephase_op = n_local;
        for (Complex& x : dephase_op) {
            x *= std::sqrt(1.0 / config.tphi[i]);
        }
        dissipator_terms.push_back(OperatorTerm{
            TermKind::Dissipator, "dephase_" + std::to_string(i),
            {i}, dephase_op, d, d});
    }

    // ALL-TO-ALL pairwise capacitive coupling (uniform g for the benchmark).
    const Index dd = d * d;
    if (config.enable_rotating_frame) {
        // Rotating frame: every coupling is time-dependent. The full strength
        // g_ij(t) = g*sqrt(w_i(t)w_j(t)/(w_i w_j)) multiplies the rotated operator
        //   cos(D t)H1 + sin(D t)H2 - cos(S t)H3 - sin(S t)H4,
        // with D = w_i - w_j (detuning, slow) and S = w_i + w_j (sum frequency, fast
        // but with tiny g-sized amplitude). At t=0 this reduces to g*(H1 - H3) = g*op.
        const TwoSiteRotatedCoupling rot =
            make_two_site_rotated_coupling_operators(d);
        for (Index i = 0; i < n_elem; ++i) {
            for (Index j = i + 1; j < n_elem; ++j) {
                const double wi0 = config.omega[i];
                const double wj0 = config.omega[j];
                const double detuning = wi0 - wj0;
                const double sum_freq = wi0 + wj0;
                const bool di = driven[i];
                const bool dj = driven[j];
                const double g0 = config.g;
                auto g_of_t =
                    [wi0, wj0, di, dj, g0, delta_omega](double t) -> double {
                        const double wi = wi0 + (di ? delta_omega(t) : 0.0);
                        const double wj = wj0 + (dj ? delta_omega(t) : 0.0);
                        return g0 * std::sqrt((wi * wj) / (wi0 * wj0));
                    };
                const std::string tag =
                    std::to_string(i) + "_" + std::to_string(j);
                time_dependent_terms.push_back(TimeDependentTerm{
                    "rf_coup_h1_" + tag, {i, j}, rot.h1, dd, dd,
                    [g_of_t, detuning](double t) {
                        return g_of_t(t) * std::cos(detuning * t);
                    }});
                time_dependent_terms.push_back(TimeDependentTerm{
                    "rf_coup_h2_" + tag, {i, j}, rot.h2, dd, dd,
                    [g_of_t, detuning](double t) {
                        return g_of_t(t) * std::sin(detuning * t);
                    }});
                // Counter-rotating terms (sum frequency omega_i+omega_j): fast but g-small.
                // Dropping them is the rotating-wave approximation.
                if (!config.drop_counter_rotating) {
                    time_dependent_terms.push_back(TimeDependentTerm{
                        "rf_coup_h3_" + tag, {i, j}, rot.h3, dd, dd,
                        [g_of_t, sum_freq](double t) {
                            return -g_of_t(t) * std::cos(sum_freq * t);
                        }});
                    time_dependent_terms.push_back(TimeDependentTerm{
                        "rf_coup_h4_" + tag, {i, j}, rot.h4, dd, dd,
                        [g_of_t, sum_freq](double t) {
                            return -g_of_t(t) * std::sin(sum_freq * t);
                        }});
                }
            }
        }
    } else {
        for (Index i = 0; i < n_elem; ++i) {
            for (Index j = i + 1; j < n_elem; ++j) {
                std::vector<Complex> coupling_term = coupling_local;
                for (Complex& x : coupling_term) {
                    x *= config.g;
                }
                hamiltonian_terms.push_back(OperatorTerm{
                    TermKind::Hamiltonian,
                    "coupling_" + std::to_string(i) + "_" + std::to_string(j),
                    {i, j}, coupling_term, dd, dd});
            }
        }
    }

    // Time-dependent gate drives: frequency modulation of selected elements.
    if (config.enable_drives) {
        // (1) Diagonal frequency-modulation Delta_omega(t)*n on each driven element.
        //     Identical in lab and rotating frames (diagonal => commutes with H_R).
        for (Index k = 0; k < n_elem; ++k) {
            if (!driven[k]) {
                continue;
            }
            TimeScalarFunction coeff =
                [delta_omega](double t) { return delta_omega(t); };
            time_dependent_terms.push_back(TimeDependentTerm{
                "drive_omega_" + std::to_string(k), {k}, n_local, d, d, coeff});
        }

        // (2) Induced coupling modulation g_lm(t) - g on every edge incident to a
        //     driven element. LAB FRAME ONLY: the rotating-frame couplings already
        //     carry the full g_ij(t) in their time-dependent coefficients.
        if (!config.enable_rotating_frame) {
            for (Index i = 0; i < n_elem; ++i) {
                for (Index j = i + 1; j < n_elem; ++j) {
                    if (!driven[i] && !driven[j]) {
                        continue;
                    }
                    const double wi0 = config.omega[i];
                    const double wj0 = config.omega[j];
                    const bool di = driven[i];
                    const bool dj = driven[j];
                    const double g_idle = config.g;
                    TimeScalarFunction coeff =
                        [wi0, wj0, di, dj, g_idle, delta_omega](double t) -> double {
                            const double wi = wi0 + (di ? delta_omega(t) : 0.0);
                            const double wj = wj0 + (dj ? delta_omega(t) : 0.0);
                            const double ratio = (wi * wj) / (wi0 * wj0);
                            return g_idle * (std::sqrt(ratio) - 1.0);
                        };
                    time_dependent_terms.push_back(TimeDependentTerm{
                        "drive_g_" + std::to_string(i) + "_" + std::to_string(j),
                        {i, j}, coupling_local, dd, dd, coeff});
                }
            }
        }
    }

    return Model{
        local_dims, hamiltonian_terms, dissipator_terms, time_dependent_terms};
}

std::vector<Index> tunable_coupler_computational_state_indices(
    Index num_qubits, Index num_couplers, Index cutoff_dim)
{
    const Index n_elem = num_qubits + num_couplers;
    const Index d = cutoff_dim;

    // Site-0-major strides: stride[i] = d^(n_elem-1-i).
    std::vector<Index> stride(n_elem, 1);
    for (Index i = n_elem; i-- > 1;) {
        stride[i - 1] = stride[i] * d;
    }

    // Hilbert index of each computational ket (qubit sites 0..num_qubits-1 in {0,1},
    // coupler sites in |0>); qubit 0 = most-significant bit of a.
    const Index num_comp = (Index(1) << num_qubits);
    std::vector<Index> ket_hilbert(num_comp, 0);
    for (Index a = 0; a < num_comp; ++a) {
        Index h = 0;
        for (Index qi = 0; qi < num_qubits; ++qi) {
            const Index bit = (a >> (num_qubits - 1 - qi)) & Index(1);
            h += bit * stride[qi];
        }
        ket_hilbert[a] = h;
    }
    return ket_hilbert;
}

std::vector<Index> tunable_coupler_computational_density_indices(
    Index num_qubits, Index num_couplers, Index cutoff_dim)
{
    const Index n_elem = num_qubits + num_couplers;
    Index D = 1;
    for (Index i = 0; i < n_elem; ++i) {
        D *= cutoff_dim;
    }
    const std::vector<Index> ket_hilbert =
        tunable_coupler_computational_state_indices(num_qubits, num_couplers, cutoff_dim);
    const Index num_comp = ket_hilbert.size();

    std::vector<Index> indices;
    indices.reserve(num_comp * num_comp);
    for (Index a = 0; a < num_comp; ++a) {
        for (Index b = 0; b < num_comp; ++b) {
            indices.push_back(ket_hilbert[a] * D + ket_hilbert[b]);
        }
    }
    return indices;
}

// Mixed-dim variants for the loaded-model handoff: same Q1=MSB, site-0-major convention as the
// tunable_coupler_* generators, but per-site strides from local_dims (the loaded model can have
// heterogeneous dims, e.g. [3,3,2,2,3,2,2,2]). Qubits are sites 0..num_qubits-1.
std::vector<Index> computational_state_indices_for_dims(
    const std::vector<Index>& local_dims, Index num_qubits)
{
    const Index n = static_cast<Index>(local_dims.size());
    std::vector<Index> stride(n, 1);
    for (Index i = n; i-- > 1;) {
        stride[i - 1] = stride[i] * local_dims[i];
    }
    const Index num_comp = (Index(1) << num_qubits);
    std::vector<Index> ket(num_comp, 0);
    for (Index a = 0; a < num_comp; ++a) {
        Index h = 0;
        for (Index qi = 0; qi < num_qubits; ++qi) {
            const Index bit = (a >> (num_qubits - 1 - qi)) & Index(1);
            h += bit * stride[qi];
        }
        ket[a] = h;
    }
    return ket;
}

std::vector<Index> computational_density_indices_for_dims(
    const std::vector<Index>& local_dims, Index num_qubits)
{
    Index D = 1;
    for (const Index d : local_dims) D *= d;
    const std::vector<Index> ket = computational_state_indices_for_dims(local_dims, num_qubits);
    std::vector<Index> out;
    out.reserve(ket.size() * ket.size());
    for (std::size_t a = 0; a < ket.size(); ++a) {
        for (std::size_t b = 0; b < ket.size(); ++b) {
            out.push_back(ket[a] * D + ket[b]);
        }
    }
    return out;
}

// Evolve a deserialized handoff Model in the LAB frame (no rotating-frame transform: the exported
// model is already lab-frame, matching the Julia sesolve/mesolve reference). The integrator is the
// usual -tc_integrator dispatch inside evolve_*_batch_cuda_ts (use -tc_integrator 1 for custom RK4).
TransmonChainBenchmarkTiming evolve_loaded_model_schrodinger(
    const Model& model_in, const std::vector<Index>& selected_indices,
    const std::vector<std::vector<Complex>>& explicit_states,
    double t0, double tfinal, int num_steps)
{
    Model model = model_in;
    model.dissipator_terms.clear();  // closed system: Schrodinger equation only
    const Solver solver = make_solver(model);
    // Explicit states (e.g. dressed comp kets) take precedence over the computational-index basis.
    std::vector<std::vector<Complex>> initial = explicit_states.empty()
        ? make_selected_basis_state_vectors(solver, selected_indices)
        : explicit_states;
    const std::size_t D = static_cast<std::size_t>(solver.layout.hilbert_dim);
    for (const std::vector<Complex>& v : initial) {
        if (v.size() != D) {
            throw std::invalid_argument("evolve_loaded_model_schrodinger: initial state length != hilbert_dim");
        }
    }
    CudaBatchExecutionContext ctx{};
    PetscCallAbort(PETSC_COMM_SELF,
        create_cuda_state_vector_execution_context(solver, initial.size(), ctx));
    TransmonChainBenchmarkTiming timing;
    timing.num_evolved_states = initial.size();
    const auto t_start = std::chrono::steady_clock::now();
    timing.final_states =
        evolve_state_vector_batch_cuda_ts(solver, initial, t0, tfinal, num_steps, ctx);
    const auto t_end = std::chrono::steady_clock::now();
    timing.wall_seconds = std::chrono::duration<double>(t_end - t_start).count();
    PetscCallAbort(PETSC_COMM_SELF, destroy_cuda_batch_execution_context(ctx));
    return timing;
}

TransmonChainBenchmarkTiming evolve_loaded_model_liouvillian(
    const Model& model, const std::vector<Index>& selected_density_indices,
    const std::vector<std::vector<Complex>>& explicit_kets,
    double t0, double tfinal, int num_steps)
{
    const Solver solver = make_solver(model);
    const std::size_t D = static_cast<std::size_t>(solver.layout.hilbert_dim);
    std::vector<std::vector<Complex>> initial;
    if (explicit_kets.empty()) {
        initial = make_selected_basis_density_states(solver, selected_density_indices);
    } else {
        // Build each |k><k| density matrix (row-major rho[i*D+j] = ket[i]*conj(ket[j])) from a ket.
        initial.reserve(explicit_kets.size());
        for (const std::vector<Complex>& ket : explicit_kets) {
            if (ket.size() != D) {
                throw std::invalid_argument("evolve_loaded_model_liouvillian: ket length != hilbert_dim");
            }
            std::vector<Complex> rho(D * D);
            for (std::size_t i = 0; i < D; ++i) {
                for (std::size_t j = 0; j < D; ++j) {
                    rho[i * D + j] = ket[i] * std::conj(ket[j]);
                }
            }
            initial.push_back(std::move(rho));
        }
    }
    CudaBatchExecutionContext ctx{};
    PetscCallAbort(PETSC_COMM_SELF,
        create_cuda_batch_execution_context(solver, initial.size(), ctx));
    PetscCallAbort(PETSC_COMM_SELF, TSSetType(ctx.ts, TSRK));
    PetscCallAbort(PETSC_COMM_SELF, TSRKSetType(ctx.ts, TSRK4));
    TSAdapt adapt = nullptr;
    PetscCallAbort(PETSC_COMM_SELF, TSGetAdapt(ctx.ts, &adapt));
    PetscCallAbort(PETSC_COMM_SELF, TSAdaptSetType(adapt, TSADAPTNONE));
    PetscCallAbort(PETSC_COMM_SELF, TSSetRHSFunction(
        ctx.ts, nullptr, ts_rhs_function_cuda_full_model_liouvillian, &ctx.rhs_ctx));
    PetscCallAbort(PETSC_COMM_SELF, TSSetMaxSteps(ctx.ts, 1000000));
    PetscCallAbort(PETSC_COMM_SELF, TSSetExactFinalTime(ctx.ts, TS_EXACTFINALTIME_MATCHSTEP));
    PetscCallAbort(PETSC_COMM_SELF, TSSetFromOptions(ctx.ts));
    TransmonChainBenchmarkTiming timing;
    timing.num_evolved_states = initial.size();
    const auto t_start = std::chrono::steady_clock::now();
    timing.final_states =
        evolve_state_batch_cuda_ts(solver, initial, t0, tfinal, num_steps, ctx);
    const auto t_end = std::chrono::steady_clock::now();
    timing.wall_seconds = std::chrono::duration<double>(t_end - t_start).count();
    PetscCallAbort(PETSC_COMM_SELF, destroy_cuda_batch_execution_context(ctx));
    return timing;
}

TransmonChainBenchmarkTiming run_tunable_coupler_cpu_benchmark(
    const TunableCouplerBenchmarkConfig& config)
{
    const Model model = build_tunable_coupler_model(config);
    const Solver solver = make_solver(model);

    const std::vector<Index> selected_indices =
        resolve_state_selection(solver, config.state_selection);
    const std::vector<std::vector<Complex>> initial_states =
        make_selected_basis_density_states(solver, selected_indices);

    CpuBenchmarkExecutionContext ctx{};
    PetscCallAbort(PETSC_COMM_SELF, create_cpu_benchmark_execution_context(solver, ctx));

    const auto t_start = std::chrono::steady_clock::now();

    TransmonChainBenchmarkTiming timing;
    timing.num_evolved_states = selected_indices.size();
    timing.final_states.resize(selected_indices.size());
    for (Index i = 0; i < initial_states.size(); ++i) {
        timing.final_states[i] = evolve_single_state_cpu_ts(
            solver, initial_states[i], config.t0, config.tfinal,
            config.batched_num_steps, ctx);
    }

    const auto t_end = std::chrono::steady_clock::now();
    timing.wall_seconds = std::chrono::duration<double>(t_end - t_start).count();

    PetscCallAbort(PETSC_COMM_SELF, destroy_cpu_benchmark_execution_context(ctx));
    return timing;
}

TransmonChainBenchmarkTiming run_tunable_coupler_cuda_benchmark(
    const TunableCouplerBenchmarkConfig& config)
{
    const Model model = build_tunable_coupler_model(config);
    const Solver solver = make_solver(model);

    const std::vector<Index> selected_indices =
        resolve_state_selection(solver, config.state_selection);
    const std::vector<std::vector<Complex>> initial_states =
        make_selected_basis_density_states(solver, selected_indices);

    CudaBatchExecutionContext ctx{};
    PetscCallAbort(
        PETSC_COMM_SELF,
        create_cuda_batch_execution_context(solver, selected_indices.size(), ctx));

    PetscCallAbort(PETSC_COMM_SELF, TSSetType(ctx.ts, TSRK));
    PetscCallAbort(PETSC_COMM_SELF, TSRKSetType(ctx.ts, TSRK4));
    TSAdapt gpu_adapt = nullptr;
    PetscCallAbort(PETSC_COMM_SELF, TSGetAdapt(ctx.ts, &gpu_adapt));
    PetscCallAbort(PETSC_COMM_SELF, TSAdaptSetType(gpu_adapt, TSADAPTNONE));
    PetscCallAbort(PETSC_COMM_SELF, TSSetRHSFunction(
        ctx.ts, nullptr, ts_rhs_function_cuda_full_model_liouvillian, &ctx.rhs_ctx));
    PetscCallAbort(PETSC_COMM_SELF, TSSetMaxSteps(ctx.ts, 1000000));
    PetscCallAbort(PETSC_COMM_SELF, TSSetExactFinalTime(ctx.ts, TS_EXACTFINALTIME_MATCHSTEP));
    PetscCallAbort(PETSC_COMM_SELF, TSSetFromOptions(ctx.ts));

    const auto t_start = std::chrono::steady_clock::now();

    TransmonChainBenchmarkTiming timing;
    timing.num_evolved_states = selected_indices.size();
    timing.final_states = evolve_state_batch_cuda_ts(
        solver, initial_states, config.t0, config.tfinal,
        config.batched_num_steps, ctx);

    const auto t_end = std::chrono::steady_clock::now();
    timing.wall_seconds = std::chrono::duration<double>(t_end - t_start).count();

    PetscCallAbort(PETSC_COMM_SELF, destroy_cuda_batch_execution_context(ctx));
    return timing;
}

TransmonChainBenchmarkTiming run_tunable_coupler_schrodinger_cuda_benchmark(
    const TunableCouplerBenchmarkConfig& config)
{
    Model model = build_tunable_coupler_model(config);
    model.dissipator_terms.clear();  // closed system: evolve the Schrodinger equation only
    const Solver solver = make_solver(model);

    std::vector<Index> selected_indices = config.state_selection.selected_state_indices;
    if (selected_indices.empty()) {
        selected_indices = tunable_coupler_computational_state_indices(
            config.num_qubits, config.num_couplers, config.cutoff_dim);
    }
    const std::vector<std::vector<Complex>> initial_states =
        make_selected_basis_state_vectors(solver, selected_indices);

    CudaBatchExecutionContext ctx{};
    PetscCallAbort(
        PETSC_COMM_SELF,
        create_cuda_state_vector_execution_context(solver, selected_indices.size(), ctx));

    const auto t_start = std::chrono::steady_clock::now();
    TransmonChainBenchmarkTiming timing;
    timing.num_evolved_states = selected_indices.size();
    timing.final_states = evolve_state_vector_batch_cuda_ts(
        solver, initial_states, config.t0, config.tfinal, config.batched_num_steps, ctx);
    const auto t_end = std::chrono::steady_clock::now();
    timing.wall_seconds = std::chrono::duration<double>(t_end - t_start).count();

    PetscCallAbort(PETSC_COMM_SELF, destroy_cuda_batch_execution_context(ctx));
    return timing;
}

int run_psi_evolve_schrodinger_selftest()
{
    TunableCouplerBenchmarkConfig cfg{};
    cfg.num_qubits = 2; cfg.num_couplers = 1; cfg.cutoff_dim = 3;
    cfg.omega = {5.0, 5.2, 4.8}; cfg.alpha = {-0.2, -0.2, -0.2};
    cfg.t1 = {1.0e4, 1.0e4, 1.0e4}; cfg.tphi = {1.0e4, 1.0e4, 1.0e4};
    cfg.g = 0.02; cfg.t0 = 0.0; cfg.tfinal = 15.0;
    cfg.enable_drives = true; cfg.driven_sites = {0};
    cfg.drive_amplitude = 0.1; cfg.drive_t_gate = 15.0;
    cfg.enable_rotating_frame = true;  // small H -> well-resolved, tight psi-vs-density gap
    const Model model = build_tunable_coupler_model(cfg);
    const Index batch = 2;
    const Index steps = 400;

    // ---- |psi> side ----
    const Solver solver = make_solver(model);
    const Index D = solver.layout.hilbert_dim;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<std::vector<Complex>> psi0(batch, std::vector<Complex>(D));
    for (Index b = 0; b < batch; ++b) {
        double n2 = 0.0;
        for (Index i = 0; i < D; ++i) { psi0[b][i] = Complex(u(rng), u(rng)); n2 += std::norm(psi0[b][i]); }
        const double inv = 1.0 / std::sqrt(n2);
        for (Index i = 0; i < D; ++i) psi0[b][i] *= inv;
    }
    CudaBatchExecutionContext ctx_psi{};
    PetscCallAbort(PETSC_COMM_SELF,
        create_cuda_state_vector_execution_context(solver, batch, ctx_psi));
    const std::vector<std::vector<Complex>> psiT =
        evolve_state_vector_batch_cuda_ts(solver, psi0, cfg.t0, cfg.tfinal, steps, ctx_psi);
    PetscCallAbort(PETSC_COMM_SELF, destroy_cuda_batch_execution_context(ctx_psi));

    // ---- closed-density oracle: rho0 = |psi0><psi0|, evolve with the validated density solver ----
    Model closed = model; closed.dissipator_terms.clear();
    const Solver solver_rho = make_solver(closed);
    std::vector<std::vector<Complex>> rho0(batch, std::vector<Complex>(D * D, Complex{0.0, 0.0}));
    for (Index b = 0; b < batch; ++b)
        for (Index r = 0; r < D; ++r)
            for (Index c = 0; c < D; ++c)
                rho0[b][r * D + c] = psi0[b][r] * std::conj(psi0[b][c]);
    CudaBatchExecutionContext ctx_rho{};
    PetscCallAbort(PETSC_COMM_SELF,
        create_cuda_batch_execution_context(solver_rho, batch, ctx_rho));
    PetscCallAbort(PETSC_COMM_SELF, TSSetType(ctx_rho.ts, TSRK));
    PetscCallAbort(PETSC_COMM_SELF, TSRKSetType(ctx_rho.ts, TSRK4));
    TSAdapt adapt = nullptr;
    PetscCallAbort(PETSC_COMM_SELF, TSGetAdapt(ctx_rho.ts, &adapt));
    PetscCallAbort(PETSC_COMM_SELF, TSAdaptSetType(adapt, TSADAPTNONE));
    const std::vector<std::vector<Complex>> rhoT =
        evolve_state_batch_cuda_ts(solver_rho, rho0, cfg.t0, cfg.tfinal, steps, ctx_rho);
    PetscCallAbort(PETSC_COMM_SELF, destroy_cuda_batch_execution_context(ctx_rho));

    double worst = 0.0, worst_norm = 0.0;
    for (Index b = 0; b < batch; ++b) {
        double n2 = 0.0;
        for (Index i = 0; i < D; ++i) n2 += std::norm(psiT[b][i]);
        worst_norm = std::max(worst_norm, std::abs(n2 - 1.0));
        for (Index r = 0; r < D; ++r)
            for (Index c = 0; c < D; ++c)
                worst = std::max(worst,
                    std::abs(rhoT[b][r * D + c] - psiT[b][r] * std::conj(psiT[b][c])));
    }
    const bool pass = (worst < 1.0e-3) && (worst_norm < 1.0e-4);
    std::cout << "PSI EVOLVE SELFTEST (2Q1C rotating, 400 steps): |psi(T)| drift=" << worst_norm
              << "  |rho_closed - |psi><psi||=" << worst
              << (pass ? "   PASS" : "   FAIL") << std::endl;
    return pass ? 0 : 1;
}

} // namespace culindblad

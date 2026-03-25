#include "culindblad/petsc_cuda_apply.hpp"

#include <petscvec.h>

#include <string>
#include <vector>

#include "culindblad/cuda_elementwise.hpp"
#include "culindblad/cutensor_contraction_desc.hpp"
#include "culindblad/cutensor_executor.hpp"
#include "culindblad/cutensor_executor_cache.hpp"
#include "culindblad/grouped_state_layout.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

namespace {

PetscErrorCode load_petsc_vec_to_grouped_host(
    const Solver& solver,
    const GroupedStateLayout& grouped_layout,
    Vec x,
    std::vector<Complex>& grouped_input)
{
    const PetscScalar* x_ptr = nullptr;
    PetscCall(VecGetArrayRead(x, &x_ptr));

    std::vector<Complex> flat_input(
        solver.layout.density_dim, Complex{0.0, 0.0});

    for (Index i = 0; i < solver.layout.density_dim; ++i) {
        flat_input[i] = reinterpret_cast<const Complex*>(x_ptr)[i];
    }

    PetscCall(VecRestoreArrayRead(x, &x_ptr));

    regroup_flat_density_to_grouped(
        grouped_layout,
        flat_input,
        grouped_input);

    return 0;
}

PetscErrorCode store_grouped_host_to_petsc_vec(
    const Solver& solver,
    const GroupedStateLayout& grouped_layout,
    const std::vector<Complex>& grouped_output,
    Vec y)
{
    PetscScalar* y_ptr = nullptr;
    PetscCall(VecGetArray(y, &y_ptr));

    std::vector<Complex> flat_output(
        solver.layout.density_dim, Complex{0.0, 0.0});

    regroup_grouped_to_flat_density(
        grouped_layout,
        grouped_output,
        flat_output);

    for (Index i = 0; i < solver.layout.density_dim; ++i) {
        reinterpret_cast<Complex*>(y_ptr)[i] = flat_output[i];
    }

    PetscCall(VecRestoreArray(y, &y_ptr));
    return 0;
}

PetscErrorCode get_or_prepare_executor(
    CuTensorExecutorCache& executor_cache,
    const std::string& cache_key,
    const CuTensorContractionDesc& contraction_desc,
    const std::string& operator_tag,
    const std::vector<Complex>& local_op,
    std::size_t grouped_bytes,
    CuTensorExecutor*& executor)
{
    const bool cache_ok =
        get_or_create_cutensor_executor(
            executor_cache,
            cache_key,
            contraction_desc,
            local_op.size() * sizeof(Complex),
            grouped_bytes,
            grouped_bytes,
            executor);

    if (!cache_ok || executor == nullptr) {
        return PETSC_ERR_LIB;
    }

    const bool op_ok =
        ensure_cutensor_executor_operator(
            *executor,
            operator_tag,
            local_op);

    if (!op_ok) {
        return PETSC_ERR_LIB;
    }

    return 0;
}

PetscErrorCode apply_grouped_cuda_vec_impl(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    CuTensorExecutorCache& executor_cache,
    const CuTensorContractionDesc& contraction_desc,
    const std::string& cache_key,
    const std::string& operator_tag,
    Vec x,
    Vec y)
{
    std::vector<Complex> grouped_input(
        grouped_layout.grouped_size, Complex{0.0, 0.0});
    std::vector<Complex> grouped_output(
        grouped_layout.grouped_size, Complex{0.0, 0.0});

    PetscCall(load_petsc_vec_to_grouped_host(
        solver,
        grouped_layout,
        x,
        grouped_input));

    CuTensorExecutor* executor = nullptr;
    PetscCall(get_or_prepare_executor(
        executor_cache,
        cache_key,
        contraction_desc,
        operator_tag,
        local_op,
        grouped_input.size() * sizeof(Complex),
        executor));

    const bool exec_ok =
        execute_cutensor_executor_with_resident_operator(
            *executor,
            grouped_input,
            grouped_output);

    if (!exec_ok) {
        return PETSC_ERR_LIB;
    }

    PetscCall(store_grouped_host_to_petsc_vec(
        solver,
        grouped_layout,
        grouped_output,
        y));

    return 0;
}

} // namespace

PetscErrorCode apply_grouped_left_cuda_vec(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    CuTensorExecutorCache& executor_cache,
    Vec x,
    Vec y)
{
    const CuTensorContractionDesc left_desc =
        make_cutensor_left_contraction_desc(target_sites, solver.model.local_dims);

    return apply_grouped_cuda_vec_impl(
        solver,
        local_op,
        target_sites,
        grouped_layout,
        executor_cache,
        left_desc,
        "petsc_grouped_left_apply",
        "petsc_grouped_left_operator",
        x,
        y);
}

PetscErrorCode apply_grouped_right_cuda_vec(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    CuTensorExecutorCache& executor_cache,
    Vec x,
    Vec y)
{
    const CuTensorContractionDesc right_desc =
        make_cutensor_right_contraction_desc(target_sites, solver.model.local_dims);

    return apply_grouped_cuda_vec_impl(
        solver,
        local_op,
        target_sites,
        grouped_layout,
        executor_cache,
        right_desc,
        "petsc_grouped_right_apply",
        "petsc_grouped_right_operator",
        x,
        y);
}

PetscErrorCode apply_grouped_commutator_cuda_vec(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    CuTensorExecutorCache& executor_cache,
    Vec x,
    Vec y)
{
    std::vector<Complex> grouped_input(
        grouped_layout.grouped_size, Complex{0.0, 0.0});
    std::vector<Complex> grouped_output(
        grouped_layout.grouped_size, Complex{0.0, 0.0});

    PetscCall(load_petsc_vec_to_grouped_host(
        solver,
        grouped_layout,
        x,
        grouped_input));

    const CuTensorContractionDesc left_desc =
        make_cutensor_left_contraction_desc(target_sites, solver.model.local_dims);
    const CuTensorContractionDesc right_desc =
        make_cutensor_right_contraction_desc(target_sites, solver.model.local_dims);

    CuTensorExecutor* left_executor = nullptr;
    CuTensorExecutor* right_executor = nullptr;
    CuTensorExecutor* combine_executor = nullptr;

    const std::size_t grouped_bytes =
        grouped_input.size() * sizeof(Complex);

    PetscCall(get_or_prepare_executor(
        executor_cache,
        "petsc_grouped_comm_left_apply",
        left_desc,
        "petsc_grouped_comm_left_operator",
        local_op,
        grouped_bytes,
        left_executor));

    PetscCall(get_or_prepare_executor(
        executor_cache,
        "petsc_grouped_comm_right_apply",
        right_desc,
        "petsc_grouped_comm_right_operator",
        local_op,
        grouped_bytes,
        right_executor));

    PetscCall(get_or_prepare_executor(
        executor_cache,
        "petsc_grouped_comm_combine_buffer",
        left_desc,
        "petsc_grouped_comm_left_operator",
        local_op,
        grouped_bytes,
        combine_executor));

    const bool left_ok =
        upload_cutensor_executor_input(*left_executor, grouped_input) &&
        execute_cutensor_executor_device(*left_executor);

    if (!left_ok) {
        return PETSC_ERR_LIB;
    }

    const bool right_ok =
        upload_cutensor_executor_input(*right_executor, grouped_input) &&
        execute_cutensor_executor_device(*right_executor);

    if (!right_ok) {
        return PETSC_ERR_LIB;
    }

    const bool combine_ok =
        launch_commutator_combine_kernel(
            left_executor->d_output,
            right_executor->d_output,
            combine_executor->d_output,
            grouped_input.size(),
            combine_executor->stream);

    if (!combine_ok) {
        return PETSC_ERR_LIB;
    }

    const bool download_ok =
        download_cutensor_executor_output(
            *combine_executor,
            grouped_output);

    if (!download_ok) {
        return PETSC_ERR_LIB;
    }

    PetscCall(store_grouped_host_to_petsc_vec(
        solver,
        grouped_layout,
        grouped_output,
        y));

    return 0;
}

PetscErrorCode apply_grouped_dissipator_cuda_vec(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& local_op_dag,
    const std::vector<Complex>& local_op_dag_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    CuTensorExecutorCache& executor_cache,
    Vec x,
    Vec y)
{
    std::vector<Complex> grouped_input(
        grouped_layout.grouped_size, Complex{0.0, 0.0});
    std::vector<Complex> grouped_output(
        grouped_layout.grouped_size, Complex{0.0, 0.0});

    PetscCall(load_petsc_vec_to_grouped_host(
        solver,
        grouped_layout,
        x,
        grouped_input));

    const CuTensorContractionDesc left_desc =
        make_cutensor_left_contraction_desc(target_sites, solver.model.local_dims);
    const CuTensorContractionDesc right_desc =
        make_cutensor_right_contraction_desc(target_sites, solver.model.local_dims);

    CuTensorExecutor* jump_left_executor = nullptr;
    CuTensorExecutor* jump_right_executor = nullptr;
    CuTensorExecutor* norm_left_executor = nullptr;
    CuTensorExecutor* norm_right_executor = nullptr;

    const std::size_t grouped_bytes =
        grouped_input.size() * sizeof(Complex);

    PetscCall(get_or_prepare_executor(
        executor_cache,
        "petsc_grouped_diss_jump_left",
        left_desc,
        "petsc_grouped_diss_L",
        local_op,
        grouped_bytes,
        jump_left_executor));

    PetscCall(get_or_prepare_executor(
        executor_cache,
        "petsc_grouped_diss_jump_right",
        right_desc,
        "petsc_grouped_diss_Ldag",
        local_op_dag,
        grouped_bytes,
        jump_right_executor));

    PetscCall(get_or_prepare_executor(
        executor_cache,
        "petsc_grouped_diss_norm_left",
        left_desc,
        "petsc_grouped_diss_LdagL",
        local_op_dag_op,
        grouped_bytes,
        norm_left_executor));

    PetscCall(get_or_prepare_executor(
        executor_cache,
        "petsc_grouped_diss_norm_right",
        right_desc,
        "petsc_grouped_diss_LdagL",
        local_op_dag_op,
        grouped_bytes,
        norm_right_executor));

    const bool jump_left_ok =
        upload_cutensor_executor_input(*jump_left_executor, grouped_input) &&
        execute_cutensor_executor_device(*jump_left_executor);

    if (!jump_left_ok) {
        return PETSC_ERR_LIB;
    }

    const bool jump_right_ok =
        copy_cutensor_executor_output_to_input(*jump_left_executor, *jump_right_executor) &&
        execute_cutensor_executor_device(*jump_right_executor);

    if (!jump_right_ok) {
        return PETSC_ERR_LIB;
    }

    const bool norm_left_ok =
        upload_cutensor_executor_input(*norm_left_executor, grouped_input) &&
        execute_cutensor_executor_device(*norm_left_executor);

    if (!norm_left_ok) {
        return PETSC_ERR_LIB;
    }

    const bool norm_right_ok =
        upload_cutensor_executor_input(*norm_right_executor, grouped_input) &&
        execute_cutensor_executor_device(*norm_right_executor);

    if (!norm_right_ok) {
        return PETSC_ERR_LIB;
    }

    const bool combine_ok =
        launch_dissipator_combine_kernel(
            jump_right_executor->d_output,
            norm_left_executor->d_output,
            norm_right_executor->d_output,
            norm_right_executor->d_output,
            grouped_input.size(),
            norm_right_executor->stream);

    if (!combine_ok) {
        return PETSC_ERR_LIB;
    }

    const bool download_ok =
        download_cutensor_executor_output(
            *norm_right_executor,
            grouped_output);

    if (!download_ok) {
        return PETSC_ERR_LIB;
    }

    PetscCall(store_grouped_host_to_petsc_vec(
        solver,
        grouped_layout,
        grouped_output,
        y));

    return 0;
}

} // namespace culindblad
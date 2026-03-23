#include "culindblad/cutensor_dissipator.hpp"

#include <vector>

#include "culindblad/cuda_elementwise.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

bool execute_cutensor_dissipator_staged(
    CuTensorExecutor& jump_left_executor,
    CuTensorExecutor& jump_right_executor,
    CuTensorExecutor& norm_left_executor,
    CuTensorExecutor& norm_right_executor,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& local_op_dag,
    const std::vector<Complex>& local_op_dag_op,
    const std::vector<Complex>& grouped_input,
    std::vector<Complex>& grouped_output)
{
    if (!upload_cutensor_executor_operator(jump_left_executor, local_op)) {
        return false;
    }

    if (!upload_cutensor_executor_input(jump_left_executor, grouped_input)) {
        return false;
    }

    if (!execute_cutensor_executor_device(jump_left_executor)) {
        return false;
    }

    if (!upload_cutensor_executor_operator(jump_right_executor, local_op_dag)) {
        return false;
    }

    if (!copy_cutensor_executor_output_to_input(jump_left_executor, jump_right_executor)) {
        return false;
    }

    if (!execute_cutensor_executor_device(jump_right_executor)) {
        return false;
    }

    if (!upload_cutensor_executor_operator(norm_left_executor, local_op_dag_op)) {
        return false;
    }

    if (!upload_cutensor_executor_input(norm_left_executor, grouped_input)) {
        return false;
    }

    if (!execute_cutensor_executor_device(norm_left_executor)) {
        return false;
    }

    if (!upload_cutensor_executor_operator(norm_right_executor, local_op_dag_op)) {
        return false;
    }

    if (!upload_cutensor_executor_input(norm_right_executor, grouped_input)) {
        return false;
    }

    if (!execute_cutensor_executor_device(norm_right_executor)) {
        return false;
    }

    if (!launch_dissipator_combine_kernel(
            jump_right_executor.d_output,
            norm_left_executor.d_output,
            norm_right_executor.d_output,
            norm_right_executor.d_output,
            grouped_input.size(),
            norm_right_executor.stream)) {
        return false;
    }

    if (!download_cutensor_executor_output(
            norm_right_executor,
            grouped_output)) {
        return false;
    }

    return true;
}

} // namespace culindblad
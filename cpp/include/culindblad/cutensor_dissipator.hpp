#pragma once

#include <vector>

#include "culindblad/cutensor_executor.hpp"
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
    std::vector<Complex>& grouped_output);

} // namespace culindblad
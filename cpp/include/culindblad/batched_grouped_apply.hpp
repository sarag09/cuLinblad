#pragma once

#include <vector>

#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

struct BatchedGroupedApplyTiming {
    double wall_seconds;
    double gpu_seconds;
    std::vector<std::vector<Complex>> output_states;
};

std::vector<std::vector<Complex>> apply_batched_grouped_left_cuda_prototype(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<std::vector<Complex>>& flat_states);

std::vector<std::vector<Complex>> apply_batched_grouped_left_cuda_device_staged(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<std::vector<Complex>>& flat_states);

BatchedGroupedApplyTiming time_batched_grouped_left_cuda_prototype(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<std::vector<Complex>>& flat_states);

BatchedGroupedApplyTiming time_batched_grouped_left_cuda_device_staged(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<std::vector<Complex>>& flat_states);

} // namespace culindblad
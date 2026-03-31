#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>

#include "culindblad/cutensor_executor.hpp"

namespace culindblad {

struct CuTensorExecutorCacheEntry {
    std::unique_ptr<CuTensorExecutor> executor;
};

struct CuTensorExecutorCache {
    std::unordered_map<std::uint64_t, CuTensorExecutorCacheEntry> entries;
};

bool get_or_create_cutensor_executor(
    CuTensorExecutorCache& cache,
    const CuTensorContractionDesc& desc,
    size_t op_bytes,
    size_t input_bytes,
    size_t output_bytes,
    CuTensorExecutor*& executor_out);

bool destroy_cutensor_executor_cache(
    CuTensorExecutorCache& cache);

} // namespace culindblad

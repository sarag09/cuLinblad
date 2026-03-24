#include "culindblad/cutensor_executor_cache.hpp"

#include <memory>
#include <string>
#include <unordered_map>

#include "culindblad/cutensor_executor.hpp"

namespace culindblad {

bool get_or_create_cutensor_executor(
    CuTensorExecutorCache& cache,
    const std::string& key,
    const CuTensorContractionDesc& desc,
    size_t op_bytes,
    size_t input_bytes,
    size_t output_bytes,
    CuTensorExecutor*& executor_out)
{
    auto it = cache.entries.find(key);
    if (it != cache.entries.end()) {
        executor_out = it->second.executor.get();
        return true;
    }

    auto executor = std::make_unique<CuTensorExecutor>();
    if (!create_cutensor_executor(
            desc,
            op_bytes,
            input_bytes,
            output_bytes,
            *executor)) {
        return false;
    }

    executor_out = executor.get();
    cache.entries.emplace(
        key,
        CuTensorExecutorCacheEntry{std::move(executor)});

    return true;
}

bool destroy_cutensor_executor_cache(
    CuTensorExecutorCache& cache)
{
    bool ok = true;

    for (auto& kv : cache.entries) {
        if (kv.second.executor) {
            if (!destroy_cutensor_executor(*kv.second.executor)) {
                ok = false;
            }
        }
    }

    cache.entries.clear();
    return ok;
}

} // namespace culindblad

#include "culindblad/cutensor_executor_cache.hpp"

#include <memory>
#include <string>
#include <unordered_map>

#include "culindblad/cutensor_executor.hpp"

namespace culindblad {

namespace {

constexpr std::size_t kMaxCachedExecutors = 6;

void touch_cache_entry(
    CuTensorExecutorCache& cache,
    CuTensorExecutorCacheEntry& entry)
{
    ++cache.generation;
    entry.last_use_generation = cache.generation;
}

bool trim_cutensor_executor_cache(
    CuTensorExecutorCache& cache)
{
    bool ok = true;

    while (cache.entries.size() > kMaxCachedExecutors) {
        auto evict_it = cache.entries.end();

        for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
            if (evict_it == cache.entries.end() ||
                it->second.last_use_generation < evict_it->second.last_use_generation) {
                evict_it = it;
            }
        }

        if (evict_it == cache.entries.end()) {
            break;
        }

        if (evict_it->second.executor &&
            !destroy_cutensor_executor(*evict_it->second.executor)) {
            ok = false;
        }

        cache.entries.erase(evict_it);
    }

    return ok;
}

} // namespace

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
        touch_cache_entry(cache, it->second);
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
    auto [inserted_it, inserted] = cache.entries.emplace(
        key,
        CuTensorExecutorCacheEntry{std::move(executor)});
    if (!inserted) {
        return false;
    }

    touch_cache_entry(cache, inserted_it->second);

    if (!trim_cutensor_executor_cache(cache)) {
        return false;
    }

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

#include "culindblad/cutensor_executor_cache.hpp"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "culindblad/cutensor_executor.hpp"

namespace culindblad {

namespace {

constexpr std::uint64_t kFnvOffsetBasis = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

template <typename T>
std::uint64_t fnv1a_append_scalar(
    std::uint64_t hash,
    const T& value)
{
    const auto* raw = reinterpret_cast<const unsigned char*>(&value);
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        hash ^= static_cast<std::uint64_t>(raw[i]);
        hash *= kFnvPrime;
    }
    return hash;
}

template <typename T>
std::uint64_t fnv1a_append_vector(
    std::uint64_t hash,
    const std::vector<T>& values)
{
    hash = fnv1a_append_scalar(hash, values.size());
    for (const T& value : values) {
        hash = fnv1a_append_scalar(hash, value);
    }
    return hash;
}

std::uint64_t make_executor_structural_key(
    CuTensorExecutorRole role,
    const CuTensorContractionDesc& desc,
    size_t op_bytes,
    size_t input_bytes,
    size_t output_bytes)
{
    std::uint64_t hash = kFnvOffsetBasis;

    hash = fnv1a_append_scalar(hash, static_cast<std::uint64_t>(role));
    hash = fnv1a_append_scalar(hash, op_bytes);
    hash = fnv1a_append_scalar(hash, input_bytes);
    hash = fnv1a_append_scalar(hash, output_bytes);
    hash = fnv1a_append_scalar(hash, desc.spec.contracted_dim);
    hash = fnv1a_append_scalar(hash, desc.spec.preserved_dim);
    hash = fnv1a_append_scalar(hash, desc.spec.passive_full_dim);
    hash = fnv1a_append_vector(hash, desc.operator_modes);
    hash = fnv1a_append_vector(hash, desc.input_modes);
    hash = fnv1a_append_vector(hash, desc.output_modes);
    hash = fnv1a_append_vector(hash, desc.operator_extents);
    hash = fnv1a_append_vector(hash, desc.operator_strides);
    hash = fnv1a_append_vector(hash, desc.input_extents);
    hash = fnv1a_append_vector(hash, desc.input_strides);
    hash = fnv1a_append_vector(hash, desc.output_extents);
    hash = fnv1a_append_vector(hash, desc.output_strides);

    return hash;
}

} // namespace

bool get_or_create_cutensor_executor(
    CuTensorExecutorCache& cache,
    CuTensorExecutorRole role,
    const CuTensorContractionDesc& desc,
    size_t op_bytes,
    size_t input_bytes,
    size_t output_bytes,
    CuTensorExecutor*& executor_out)
{
    const std::uint64_t key =
        make_executor_structural_key(
            role,
            desc,
            op_bytes,
            input_bytes,
            output_bytes);

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

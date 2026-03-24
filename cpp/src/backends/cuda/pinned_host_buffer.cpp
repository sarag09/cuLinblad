#include "culindblad/pinned_host_buffer.hpp"

#include <cstddef>
#include <cuda_runtime.h>

#include "culindblad/types.hpp"

namespace culindblad {

bool create_pinned_complex_buffer(
    std::size_t size,
    PinnedComplexBuffer& buffer)
{
    buffer.data = nullptr;
    buffer.size = size;

    if (cudaMallocHost(reinterpret_cast<void**>(&buffer.data),
                       size * sizeof(Complex)) != cudaSuccess) {
        buffer.data = nullptr;
        buffer.size = 0;
        return false;
    }

    return true;
}

bool destroy_pinned_complex_buffer(
    PinnedComplexBuffer& buffer)
{
    bool ok = true;

    if (buffer.data != nullptr) {
        if (cudaFreeHost(buffer.data) != cudaSuccess) {
            ok = false;
        }
    }

    buffer.data = nullptr;
    buffer.size = 0;
    return ok;
}

} // namespace culindblad

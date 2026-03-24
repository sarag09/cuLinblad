#pragma once

#include <cstddef>

#include "culindblad/types.hpp"

namespace culindblad {

struct PinnedComplexBuffer {
    Complex* data;
    std::size_t size;
};

bool create_pinned_complex_buffer(
    std::size_t size,
    PinnedComplexBuffer& buffer);

bool destroy_pinned_complex_buffer(
    PinnedComplexBuffer& buffer);

} // namespace culindblad

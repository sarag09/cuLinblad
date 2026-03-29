#pragma once

#include <vector>

#include "culindblad/grouped_state_layout.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

struct BatchedGroupedLayout {
    GroupedStateLayout single_layout;
    Index batch_size;
    Index per_state_grouped_size;
    Index total_grouped_size;
};

BatchedGroupedLayout make_batched_grouped_layout(
    const GroupedStateLayout& single_layout,
    Index batch_size);

void pack_flat_batch_to_grouped_batch(
    const BatchedGroupedLayout& batched_layout,
    const std::vector<std::vector<Complex>>& flat_states,
    std::vector<Complex>& grouped_batch);

void unpack_grouped_batch_to_flat_batch(
    const BatchedGroupedLayout& batched_layout,
    const std::vector<Complex>& grouped_batch,
    std::vector<std::vector<Complex>>& flat_states);

} // namespace culindblad

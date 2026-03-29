#include "culindblad/batched_grouped_layout.hpp"

#include <stdexcept>
#include <vector>

#include "culindblad/grouped_state_layout.hpp"

namespace culindblad {

BatchedGroupedLayout make_batched_grouped_layout(
    const GroupedStateLayout& single_layout,
    Index batch_size)
{
    if (batch_size == 0) {
        throw std::invalid_argument(
            "make_batched_grouped_layout: batch_size must be > 0");
    }

    BatchedGroupedLayout batched;
    batched.single_layout = single_layout;
    batched.batch_size = batch_size;
    batched.per_state_grouped_size = single_layout.grouped_size;
    batched.total_grouped_size = batch_size * single_layout.grouped_size;
    return batched;
}

void pack_flat_batch_to_grouped_batch(
    const BatchedGroupedLayout& batched_layout,
    const std::vector<std::vector<Complex>>& flat_states,
    std::vector<Complex>& grouped_batch)
{
    if (flat_states.size() != batched_layout.batch_size) {
        throw std::invalid_argument(
            "pack_flat_batch_to_grouped_batch: flat_states size mismatch");
    }

    grouped_batch.assign(
        batched_layout.total_grouped_size,
        Complex{0.0, 0.0});

    const Index flat_size =
        batched_layout.single_layout.hilbert_dim *
        batched_layout.single_layout.hilbert_dim;
    const Index grouped_size = batched_layout.single_layout.grouped_size;

    for (Index batch = 0; batch < batched_layout.batch_size; ++batch) {
        if (flat_states[batch].size() != flat_size) {
            throw std::invalid_argument(
                "pack_flat_batch_to_grouped_batch: individual flat state size mismatch");
        }

        std::vector<Complex> grouped_single(grouped_size, Complex{0.0, 0.0});

        regroup_flat_density_to_grouped(
            batched_layout.single_layout,
            flat_states[batch],
            grouped_single);

        const Index offset = batch * grouped_size;
        for (Index i = 0; i < grouped_size; ++i) {
            grouped_batch[offset + i] = grouped_single[i];
        }
    }
}

void unpack_grouped_batch_to_flat_batch(
    const BatchedGroupedLayout& batched_layout,
    const std::vector<Complex>& grouped_batch,
    std::vector<std::vector<Complex>>& flat_states)
{
    if (grouped_batch.size() != batched_layout.total_grouped_size) {
        throw std::invalid_argument(
            "unpack_grouped_batch_to_flat_batch: grouped_batch size mismatch");
    }

    const Index flat_size =
        batched_layout.single_layout.hilbert_dim *
        batched_layout.single_layout.hilbert_dim;
    const Index grouped_size = batched_layout.single_layout.grouped_size;

    flat_states.resize(batched_layout.batch_size);

    for (Index batch = 0; batch < batched_layout.batch_size; ++batch) {
        std::vector<Complex> grouped_single(grouped_size, Complex{0.0, 0.0});
        const Index offset = batch * grouped_size;

        for (Index i = 0; i < grouped_size; ++i) {
            grouped_single[i] = grouped_batch[offset + i];
        }

        flat_states[batch].assign(flat_size, Complex{0.0, 0.0});

        regroup_grouped_to_flat_density(
            batched_layout.single_layout,
            grouped_single,
            flat_states[batch]);
    }
}

} // namespace culindblad
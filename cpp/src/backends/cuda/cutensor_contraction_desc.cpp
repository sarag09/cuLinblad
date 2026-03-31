#include "culindblad/cutensor_contraction_desc.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include "culindblad/k_site_contraction_api.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

namespace {

std::vector<int64_t> make_row_major_strides(
    const std::vector<int64_t>& extents)
{
    std::vector<int64_t> strides(extents.size(), 1);

    if (extents.empty()) {
        return strides;
    }

    strides.back() = 1;
    for (Index i = extents.size() - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * extents[i];
    }

    return strides;
}

void set_default_row_major_strides(
    CuTensorContractionDesc& desc)
{
    desc.operator_strides = make_row_major_strides(desc.operator_extents);
    desc.input_strides = make_row_major_strides(desc.input_extents);
    desc.output_strides = make_row_major_strides(desc.output_extents);
}

} // namespace

CuTensorContractionDesc make_cutensor_left_contraction_desc(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims)
{
    CuTensorContractionDesc desc;
    desc.spec = make_grouped_left_contraction_spec(target_sites, local_dims);

    desc.debug_name = "cutensor_grouped_left_action";

    desc.operator_modes = {0, 1};
    desc.input_modes = {1, 2, 3, 4};
    desc.output_modes = {0, 2, 3, 4};

    desc.operator_extents = {
        static_cast<int64_t>(desc.spec.contracted_dim),
        static_cast<int64_t>(desc.spec.contracted_dim)
    };

    desc.input_extents = {
        static_cast<int64_t>(desc.spec.roles.desc.view.ket_target_dim),
        static_cast<int64_t>(desc.spec.roles.desc.view.ket_complement_dim),
        static_cast<int64_t>(desc.spec.roles.desc.view.bra_target_dim),
        static_cast<int64_t>(desc.spec.roles.desc.view.bra_complement_dim)
    };

    desc.output_extents = {
        static_cast<int64_t>(desc.spec.roles.desc.view.ket_target_dim),
        static_cast<int64_t>(desc.spec.roles.desc.view.ket_complement_dim),
        static_cast<int64_t>(desc.spec.roles.desc.view.bra_target_dim),
        static_cast<int64_t>(desc.spec.roles.desc.view.bra_complement_dim)
    };

    set_default_row_major_strides(desc);

    return desc;
}

CuTensorContractionDesc make_batched_cutensor_left_contraction_desc(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims,
    Index batch_size)
{
    CuTensorContractionDesc desc =
        make_cutensor_left_contraction_desc(target_sites, local_dims);

    desc.debug_name = "cutensor_grouped_left_action_batched";
    desc.input_modes = {5, 1, 2, 3, 4};
    desc.output_modes = {5, 0, 2, 3, 4};

    desc.input_extents.insert(
        desc.input_extents.begin(),
        static_cast<int64_t>(batch_size));
    desc.output_extents.insert(
        desc.output_extents.begin(),
        static_cast<int64_t>(batch_size));
    set_default_row_major_strides(desc);

    return desc;
}

CuTensorContractionDesc make_cutensor_right_contraction_desc(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims)
{
    CuTensorContractionDesc desc;
    desc.spec = make_grouped_right_contraction_spec(target_sites, local_dims);

    desc.debug_name = "cutensor_grouped_right_action";

    desc.operator_modes = {1, 0};
    desc.input_modes = {2, 3, 1, 4};
    desc.output_modes = {2, 3, 0, 4};

    desc.operator_extents = {
        static_cast<int64_t>(desc.spec.contracted_dim),
        static_cast<int64_t>(desc.spec.contracted_dim)
    };

    desc.input_extents = {
        static_cast<int64_t>(desc.spec.roles.desc.view.ket_target_dim),
        static_cast<int64_t>(desc.spec.roles.desc.view.ket_complement_dim),
        static_cast<int64_t>(desc.spec.roles.desc.view.bra_target_dim),
        static_cast<int64_t>(desc.spec.roles.desc.view.bra_complement_dim)
    };

    desc.output_extents = {
        static_cast<int64_t>(desc.spec.roles.desc.view.ket_target_dim),
        static_cast<int64_t>(desc.spec.roles.desc.view.ket_complement_dim),
        static_cast<int64_t>(desc.spec.roles.desc.view.bra_target_dim),
        static_cast<int64_t>(desc.spec.roles.desc.view.bra_complement_dim)
    };

    set_default_row_major_strides(desc);

    return desc;
}

CuTensorContractionDesc make_batched_cutensor_right_contraction_desc(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims,
    Index batch_size)
{
    CuTensorContractionDesc desc =
        make_cutensor_right_contraction_desc(target_sites, local_dims);

    desc.debug_name = "cutensor_grouped_right_action_batched";
    desc.input_modes = {5, 2, 3, 1, 4};
    desc.output_modes = {5, 2, 3, 0, 4};

    desc.input_extents.insert(
        desc.input_extents.begin(),
        static_cast<int64_t>(batch_size));
    desc.output_extents.insert(
        desc.output_extents.begin(),
        static_cast<int64_t>(batch_size));
    set_default_row_major_strides(desc);

    return desc;
}

} // namespace culindblad

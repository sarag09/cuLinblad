#include "culindblad/cutensor_contraction_desc.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include "culindblad/k_site_contraction_api.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

CuTensorContractionDesc make_cutensor_left_contraction_desc(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims,
    Index batch_size)
{
    CuTensorContractionDesc desc;
    desc.spec = make_grouped_left_contraction_spec(target_sites, local_dims);

    desc.debug_name = "cutensor_grouped_left_action";
    const int32_t batch_mode = 5;

    desc.operator_modes = {0, 1};
    const bool batched = batch_size > 1;
    desc.input_modes = batched ? std::vector<int32_t>{batch_mode, 1, 2, 3, 4}
                               : std::vector<int32_t>{1, 2, 3, 4};
    desc.output_modes = batched ? std::vector<int32_t>{batch_mode, 0, 2, 3, 4}
                                : std::vector<int32_t>{0, 2, 3, 4};

    desc.operator_extents = {
        static_cast<int64_t>(desc.spec.contracted_dim),
        static_cast<int64_t>(desc.spec.contracted_dim)
    };

    if (batched) {
        desc.input_extents = {
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(desc.spec.roles.desc.view.ket_target_dim),
            static_cast<int64_t>(desc.spec.roles.desc.view.ket_complement_dim),
            static_cast<int64_t>(desc.spec.roles.desc.view.bra_target_dim),
            static_cast<int64_t>(desc.spec.roles.desc.view.bra_complement_dim)
        };
        desc.output_extents = {
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(desc.spec.roles.desc.view.ket_target_dim),
            static_cast<int64_t>(desc.spec.roles.desc.view.ket_complement_dim),
            static_cast<int64_t>(desc.spec.roles.desc.view.bra_target_dim),
            static_cast<int64_t>(desc.spec.roles.desc.view.bra_complement_dim)
        };
    } else {
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
    }

    return desc;
}

CuTensorContractionDesc make_cutensor_right_contraction_desc(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims,
    Index batch_size)
{
    CuTensorContractionDesc desc;
    desc.spec = make_grouped_right_contraction_spec(target_sites, local_dims);

    desc.debug_name = "cutensor_grouped_right_action";
    const int32_t batch_mode = 5;

    desc.operator_modes = {1, 0};
    const bool batched = batch_size > 1;
    desc.input_modes = batched ? std::vector<int32_t>{batch_mode, 2, 3, 1, 4}
                               : std::vector<int32_t>{2, 3, 1, 4};
    desc.output_modes = batched ? std::vector<int32_t>{batch_mode, 2, 3, 0, 4}
                                : std::vector<int32_t>{2, 3, 0, 4};

    desc.operator_extents = {
        static_cast<int64_t>(desc.spec.contracted_dim),
        static_cast<int64_t>(desc.spec.contracted_dim)
    };

    if (batched) {
        desc.input_extents = {
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(desc.spec.roles.desc.view.ket_target_dim),
            static_cast<int64_t>(desc.spec.roles.desc.view.ket_complement_dim),
            static_cast<int64_t>(desc.spec.roles.desc.view.bra_target_dim),
            static_cast<int64_t>(desc.spec.roles.desc.view.bra_complement_dim)
        };
        desc.output_extents = {
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(desc.spec.roles.desc.view.ket_target_dim),
            static_cast<int64_t>(desc.spec.roles.desc.view.ket_complement_dim),
            static_cast<int64_t>(desc.spec.roles.desc.view.bra_target_dim),
            static_cast<int64_t>(desc.spec.roles.desc.view.bra_complement_dim)
        };
    } else {
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
    }

    return desc;
}

} // namespace culindblad

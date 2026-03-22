#pragma once

#include <cutensor.h>

#include "culindblad/cutensor_operation_desc.hpp"

namespace culindblad {

struct CuTensorPlanBundle {
    CuTensorOperationDesc op_bundle;
    cutensorPlanPreference_t preference;
    cutensorPlan_t plan;
    uint64_t workspace_size;
};

bool create_cutensor_plan(
    const CuTensorContractionDesc& desc,
    CuTensorPlanBundle& plan_bundle);

bool destroy_cutensor_plan(
    CuTensorPlanBundle& plan_bundle);

} // namespace culindblad

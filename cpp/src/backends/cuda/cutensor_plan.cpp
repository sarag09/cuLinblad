#include "culindblad/cutensor_plan.hpp"

#include <cutensor.h>

namespace culindblad {

bool create_cutensor_plan(
    const CuTensorContractionDesc& desc,
    CuTensorPlanBundle& plan_bundle)
{
    if (!create_cutensor_operation_desc(desc, plan_bundle.op_bundle)) {
        return false;
    }

    auto& handle = plan_bundle.op_bundle.tensor_descs.handle;
    auto& op_desc = plan_bundle.op_bundle.op_desc;

    const cutensorStatus_t pref_status =
        cutensorCreatePlanPreference(
            handle,
            &plan_bundle.preference,
            CUTENSOR_ALGO_DEFAULT,
            CUTENSOR_JIT_MODE_NONE);

    if (pref_status != CUTENSOR_STATUS_SUCCESS) {
        destroy_cutensor_operation_desc(plan_bundle.op_bundle);
        return false;
    }

    const cutensorStatus_t ws_status =
        cutensorEstimateWorkspaceSize(
            handle,
            op_desc,
            plan_bundle.preference,
            CUTENSOR_WORKSPACE_DEFAULT,
            &plan_bundle.workspace_size);

    if (ws_status != CUTENSOR_STATUS_SUCCESS) {
        cutensorDestroyPlanPreference(plan_bundle.preference);
        destroy_cutensor_operation_desc(plan_bundle.op_bundle);
        return false;
    }

    const cutensorStatus_t plan_status =
        cutensorCreatePlan(
            handle,
            &plan_bundle.plan,
            op_desc,
            plan_bundle.preference,
            plan_bundle.workspace_size);

    if (plan_status != CUTENSOR_STATUS_SUCCESS) {
        cutensorDestroyPlanPreference(plan_bundle.preference);
        destroy_cutensor_operation_desc(plan_bundle.op_bundle);
        return false;
    }

    return true;
}

bool destroy_cutensor_plan(
    CuTensorPlanBundle& plan_bundle)
{
    bool ok = true;

    if (cutensorDestroyPlan(plan_bundle.plan) != CUTENSOR_STATUS_SUCCESS) {
        ok = false;
    }

    if (cutensorDestroyPlanPreference(plan_bundle.preference) != CUTENSOR_STATUS_SUCCESS) {
        ok = false;
    }

    if (!destroy_cutensor_operation_desc(plan_bundle.op_bundle)) {
        ok = false;
    }

    return ok;
}

} // namespace culindblad
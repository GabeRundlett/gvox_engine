#pragma once

#include <shared/core.inl>

DAXA_DECL_TASK_USES_BEGIN(CalculateReprojectionMapComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(vs_normal_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(depth_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(prev_depth_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(velocity_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(dst_image_id, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()

#if defined(__cplusplus)

struct CalculateReprojectionMapComputeTaskState {
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    CalculateReprojectionMapComputeTaskState(daxa::PipelineManager &pipeline_manager) {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"calculate_reprojection_map.comp.glsl"},
                .compile_options = {.defines = {{"CALCULATE_REPROJECTION_MAP_COMPUTE", "1"}}},
            },
            .name = "calculate_reprojection_map",
        });
        if (compile_result.is_err()) {
            AppUi::Console::s_instance->add_log(compile_result.message());
            return;
        }
        pipeline = compile_result.value();
        if (!compile_result.value()->is_valid()) {
            AppUi::Console::s_instance->add_log(compile_result.message());
        }
    }
    auto pipeline_is_valid() -> bool { return pipeline && pipeline->is_valid(); }

    void record_commands(daxa::CommandList &cmd_list, u32vec2 render_size) {
        if (!pipeline_is_valid()) {
            return;
        }
        cmd_list.set_pipeline(*pipeline);
        assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
        cmd_list.dispatch(render_size.x / 8, render_size.y / 8);
    }
};

struct CalculateReprojectionMapComputeTask : CalculateReprojectionMapComputeUses {
    CalculateReprojectionMapComputeTaskState *state;
    u32vec2 render_size;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        auto const &image_info = ti.get_device().info_image(uses.dst_image_id.image());
        state->record_commands(cmd_list, {image_info.size.x, image_info.size.y});
    }
};

#endif

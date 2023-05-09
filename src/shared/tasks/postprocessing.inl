#pragma once

#include "../core.inl"

DAXA_INL_TASK_USE_BEGIN(PostprocessingComputeUses, DAXA_CBUFFER_SLOT0)
DAXA_INL_TASK_USE_BUFFER(settings, daxa_BufferPtr(GpuSettings), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_IMAGE(render_col_image_id, daxa_RWImage2Df32, COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_IMAGE(final_image_id, daxa_RWImage2Df32, COMPUTE_SHADER_WRITE)
DAXA_INL_TASK_USE_END()

#if defined(__cplusplus)

struct PostprocessingComputeTaskState {
    daxa::PipelineManager &pipeline_manager;
    AppUi &ui;
    u32vec2 &render_size;
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    void compile_pipeline() {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"postprocessing.comp.glsl"},
                .compile_options = {.defines = {{"POSTPROCESSING_COMPUTE", "1"}}},
            },
            .name = "postprocessing",
        });
        if (compile_result.is_err()) {
            ui.console.add_log(compile_result.to_string());
            return;
        }
        pipeline = compile_result.value();
    }

    PostprocessingComputeTaskState(daxa::PipelineManager &a_pipeline_manager, AppUi &a_ui, u32vec2 &a_render_size) : pipeline_manager{a_pipeline_manager}, ui{a_ui}, render_size{a_render_size} {}

    void record_commands(daxa::CommandList &cmd_list) {
        if (!pipeline) {
            compile_pipeline();
            if (!pipeline)
                return;
        }
        cmd_list.set_pipeline(*pipeline);
        cmd_list.dispatch((render_size.x + 7) / 8, (render_size.y + 7) / 8);
    }
};

struct PostprocessingComputeTask : PostprocessingComputeUses {
    PostprocessingComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_constant_buffer(ti.uses.constant_buffer_set_info());
        state->record_commands(cmd_list);
    }
};

#endif

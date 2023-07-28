#pragma once

#include <shared/core.inl>

DAXA_DECL_TASK_USES_BEGIN(DownscaleComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(src_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(dst_image_id, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()

#if defined(__cplusplus)

struct DownscaleComputeTaskState {
    daxa::PipelineManager &pipeline_manager;
    AppUi &ui;
    u32vec2 &render_size;
    std::shared_ptr<daxa::ComputePipeline> pipeline;
    std::vector<daxa::ShaderDefine> defines;

    void compile_pipeline() {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"downscale.comp.glsl"},
                .compile_options = {.defines = defines},
            },
            .name = "downscale",
        });
        if (compile_result.is_err()) {
            ui.console.add_log(compile_result.message());
            return;
        }
        pipeline = compile_result.value();
        if (!compile_result.value()->is_valid()) {
            ui.console.add_log(compile_result.message());
        }
    }

    DownscaleComputeTaskState(
        daxa::PipelineManager &a_pipeline_manager, AppUi &a_ui, u32vec2 &a_render_size,
        std::vector<daxa::ShaderDefine> &&extra_defines)
        : pipeline_manager{a_pipeline_manager}, ui{a_ui}, render_size{a_render_size}, defines{extra_defines} {
        this->defines.push_back({"DOWNSCALE_COMPUTE", "1"});
        compile_pipeline();
    }
    auto pipeline_is_valid() -> bool { return pipeline && pipeline->is_valid(); }

    void record_commands(daxa::CommandList &cmd_list) {
        if (!pipeline_is_valid()) {
            return;
        }
        cmd_list.set_pipeline(*pipeline);
        constexpr auto SCL = 8 * SHADING_SCL;
        assert((render_size.x % SCL) == 0 && (render_size.y % SCL) == 0);
        cmd_list.dispatch(render_size.x / SCL, render_size.y / SCL);
    }
};

struct DownscaleComputeTask : DownscaleComputeUses {
    DownscaleComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        state->record_commands(cmd_list);
    }
};

#endif

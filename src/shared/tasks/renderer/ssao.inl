#pragma once

#include <shared/core.inl>

#if SSAO_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(SsaoComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(vs_normal_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(depth_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(ssao_image_id, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif

#if SSAO_SPATIAL_FILTER_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(SsaoSpatialFilterComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(vs_normal_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(depth_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(src_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(dst_image_id, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif

#if SSAO_UPSAMPLE_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(SsaoUpscaleComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(g_buffer_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(depth_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(src_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(dst_image_id, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif

#if SSAO_TEMPORAL_FILTER_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(SsaoTemporalFilterComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(reprojection_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(history_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(src_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(dst_image_id, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif

struct SsaoTemporalFilterComputePush {
    daxa_SamplerId history_sampler;
};

#if defined(__cplusplus)

struct SsaoComputeTaskState {
    daxa::PipelineManager &pipeline_manager;
    AppUi &ui;
    u32vec2 &render_size;
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    void compile_pipeline() {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"ssao.comp.glsl"},
                .compile_options = {.defines = {{"SSAO_COMPUTE", "1"}}},
            },
            .name = "ssao",
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

    SsaoComputeTaskState(daxa::PipelineManager &a_pipeline_manager, AppUi &a_ui, u32vec2 &a_render_size) : pipeline_manager{a_pipeline_manager}, ui{a_ui}, render_size{a_render_size} { compile_pipeline(); }
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

struct SsaoSpatialFilterComputeTaskState {
    daxa::PipelineManager &pipeline_manager;
    AppUi &ui;
    u32vec2 &render_size;
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    void compile_pipeline() {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"ssao.comp.glsl"},
                .compile_options = {.defines = {{"SSAO_SPATIAL_FILTER_COMPUTE", "1"}}},
            },
            .name = "spatial_filter",
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

    SsaoSpatialFilterComputeTaskState(daxa::PipelineManager &a_pipeline_manager, AppUi &a_ui, u32vec2 &a_render_size) : pipeline_manager{a_pipeline_manager}, ui{a_ui}, render_size{a_render_size} { compile_pipeline(); }
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

struct SsaoUpscaleComputeTaskState {
    daxa::PipelineManager &pipeline_manager;
    AppUi &ui;
    u32vec2 &render_size;
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    void compile_pipeline() {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"ssao.comp.glsl"},
                .compile_options = {.defines = {{"SSAO_UPSAMPLE_COMPUTE", "1"}}},
            },
            .name = "ssao_upscale",
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

    SsaoUpscaleComputeTaskState(daxa::PipelineManager &a_pipeline_manager, AppUi &a_ui, u32vec2 &a_render_size) : pipeline_manager{a_pipeline_manager}, ui{a_ui}, render_size{a_render_size} { compile_pipeline(); }
    auto pipeline_is_valid() -> bool { return pipeline && pipeline->is_valid(); }

    void record_commands(daxa::CommandList &cmd_list) {
        if (!pipeline_is_valid()) {
            return;
        }
        cmd_list.set_pipeline(*pipeline);
        constexpr auto SCL = 8;
        assert((render_size.x % SCL) == 0 && (render_size.y % SCL) == 0);
        cmd_list.dispatch(render_size.x / SCL, render_size.y / SCL);
    }
};

struct SsaoTemporalFilterComputeTaskState {
    daxa::PipelineManager &pipeline_manager;
    AppUi &ui;
    daxa::SamplerId &history_sampler;
    u32vec2 &render_size;
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    void compile_pipeline() {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"ssao.comp.glsl"},
                .compile_options = {.defines = {{"SSAO_TEMPORAL_FILTER_COMPUTE", "1"}}},
            },
            .push_constant_size = sizeof(SsaoTemporalFilterComputePush),
            .name = "ssao_temporal_filter",
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

    SsaoTemporalFilterComputeTaskState(daxa::PipelineManager &a_pipeline_manager, AppUi &a_ui, daxa::SamplerId &a_sampler, u32vec2 &a_render_size) : pipeline_manager{a_pipeline_manager}, ui{a_ui}, history_sampler{a_sampler}, render_size{a_render_size} { compile_pipeline(); }
    auto pipeline_is_valid() -> bool { return pipeline && pipeline->is_valid(); }

    void record_commands(daxa::CommandList &cmd_list) {
        if (!pipeline_is_valid()) {
            return;
        }
        cmd_list.set_pipeline(*pipeline);
        cmd_list.push_constant(SsaoTemporalFilterComputePush{
            .history_sampler = history_sampler,
        });
        constexpr auto SCL = 8;
        assert((render_size.x % SCL) == 0 && (render_size.y % SCL) == 0);
        cmd_list.dispatch(render_size.x / SCL, render_size.y / SCL);
    }
};

struct SsaoComputeTask : SsaoComputeUses {
    SsaoComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        state->record_commands(cmd_list);
    }
};

struct SsaoSpatialFilterComputeTask : SsaoSpatialFilterComputeUses {
    SsaoSpatialFilterComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        state->record_commands(cmd_list);
    }
};

struct SsaoUpscaleComputeTask : SsaoUpscaleComputeUses {
    SsaoUpscaleComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        state->record_commands(cmd_list);
    }
};

struct SsaoTemporalFilterComputeTask : SsaoTemporalFilterComputeUses {
    SsaoTemporalFilterComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        state->record_commands(cmd_list);
    }
};

struct SsaoRenderer {
    PingPongImage ssao_image;

    void record(daxa::TaskGraph &task_graph) {
    }
};

#endif

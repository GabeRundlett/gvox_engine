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
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    SsaoComputeTaskState(daxa::PipelineManager &pipeline_manager) {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"ssao.comp.glsl"},
                .compile_options = {.defines = {{"SSAO_COMPUTE", "1"}}},
            },
            .name = "ssao",
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

struct SsaoSpatialFilterComputeTaskState {
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    SsaoSpatialFilterComputeTaskState(daxa::PipelineManager &pipeline_manager) {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"ssao.comp.glsl"},
                .compile_options = {.defines = {{"SSAO_SPATIAL_FILTER_COMPUTE", "1"}}},
            },
            .name = "spatial_filter",
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

struct SsaoUpscaleComputeTaskState {
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    SsaoUpscaleComputeTaskState(daxa::PipelineManager &pipeline_manager) {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"ssao.comp.glsl"},
                .compile_options = {.defines = {{"SSAO_UPSAMPLE_COMPUTE", "1"}}},
            },
            .name = "ssao_upscale",
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

struct SsaoTemporalFilterComputeTaskState {
    daxa::SamplerId &history_sampler;
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    SsaoTemporalFilterComputeTaskState(daxa::PipelineManager &pipeline_manager, daxa::SamplerId &a_sampler) : history_sampler{a_sampler} {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"ssao.comp.glsl"},
                .compile_options = {.defines = {{"SSAO_TEMPORAL_FILTER_COMPUTE", "1"}}},
            },
            .push_constant_size = sizeof(SsaoTemporalFilterComputePush),
            .name = "ssao_temporal_filter",
        });
        if (compile_result.is_err()) {
            AppUi::Console::s_instance->add_log(compile_result.message());
            return;
        }
        pipeline = compile_result.value();
        if (!compile_result.value()->is_valid()) {
            AppUi::Console::s_instance->add_log(compile_result.message());
        };
    }
    auto pipeline_is_valid() -> bool { return pipeline && pipeline->is_valid(); }

    void record_commands(daxa::CommandList &cmd_list, u32vec2 render_size) {
        if (!pipeline_is_valid()) {
            return;
        }
        cmd_list.set_pipeline(*pipeline);
        cmd_list.push_constant(SsaoTemporalFilterComputePush{
            .history_sampler = history_sampler,
        });
        assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
        cmd_list.dispatch(render_size.x / 8, render_size.y / 8);
    }
};

struct SsaoComputeTask : SsaoComputeUses {
    SsaoComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        auto const &image_info = ti.get_device().info_image(uses.ssao_image_id.image());
        state->record_commands(cmd_list, {image_info.size.x, image_info.size.y});
    }
};

struct SsaoSpatialFilterComputeTask : SsaoSpatialFilterComputeUses {
    SsaoSpatialFilterComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        auto const &image_info = ti.get_device().info_image(uses.dst_image_id.image());
        state->record_commands(cmd_list, {image_info.size.x, image_info.size.y});
    }
};

struct SsaoUpscaleComputeTask : SsaoUpscaleComputeUses {
    SsaoUpscaleComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        auto const &image_info = ti.get_device().info_image(uses.dst_image_id.image());
        state->record_commands(cmd_list, {image_info.size.x, image_info.size.y});
    }
};

struct SsaoTemporalFilterComputeTask : SsaoTemporalFilterComputeUses {
    SsaoTemporalFilterComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        auto const &image_info = ti.get_device().info_image(uses.dst_image_id.image());
        state->record_commands(cmd_list, {image_info.size.x, image_info.size.y});
    }
};

struct SsaoRenderer {
    PingPongImage ssao_image;

    void record(daxa::TaskGraph &task_graph) {
    }
};

#endif

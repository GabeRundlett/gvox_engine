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
    PingPongImage ping_pong_ssao_image;
    SsaoComputeTaskState ssao_task_state;
    SsaoSpatialFilterComputeTaskState ssao_spatial_filter_task_state;
    SsaoUpscaleComputeTaskState ssao_upscale_task_state;
    SsaoTemporalFilterComputeTaskState ssao_temporal_filter_task_state;

    SsaoRenderer(daxa::PipelineManager &pipeline_manager, daxa::SamplerId &a_sampler)
        : ssao_task_state{pipeline_manager},
          ssao_spatial_filter_task_state{pipeline_manager},
          ssao_upscale_task_state{pipeline_manager},
          ssao_temporal_filter_task_state{pipeline_manager, a_sampler} {
    }

    void next_frame() {
        ping_pong_ssao_image.task_resources.output_image.swap_images(ping_pong_ssao_image.task_resources.history_image);
    }

    auto render(RecordContext &record_ctx, GbufferDepth &gbuffer_depth, daxa::TaskImageView reprojection_map) -> daxa::TaskImageView {
        auto scaled_depth_image = gbuffer_depth.get_downscaled_depth(record_ctx);
        auto scaled_view_normal_image = gbuffer_depth.get_downscaled_view_normal(record_ctx);
        ping_pong_ssao_image = PingPongImage{};
        auto [ssao_image, prev_ssao_image] = ping_pong_ssao_image.get(
            record_ctx.device,
            {
                .format = daxa::Format::R16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC,
                .name = "ssao_image",
            });
        record_ctx.task_graph.use_persistent_image(ssao_image);
        record_ctx.task_graph.use_persistent_image(prev_ssao_image);
        auto ssao_image0 = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16_SFLOAT,
            .size = {record_ctx.render_resolution.x / SHADING_SCL, record_ctx.render_resolution.y / SHADING_SCL, 1},
            .name = "ssao_image0",
        });
        auto ssao_image1 = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16_SFLOAT,
            .size = {record_ctx.render_resolution.x / SHADING_SCL, record_ctx.render_resolution.y / SHADING_SCL, 1},
            .name = "ssao_image1",
        });
        auto ssao_image2 = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "ssao_image2",
        });
        record_ctx.task_graph.add_task(SsaoComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .globals = record_ctx.task_globals_buffer,
                    .vs_normal_image_id = scaled_view_normal_image,
                    .depth_image_id = scaled_depth_image,
                    .ssao_image_id = ssao_image0,
                },
            },
            &ssao_task_state,
        });
        record_ctx.task_graph.add_task(SsaoSpatialFilterComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .vs_normal_image_id = scaled_view_normal_image,
                    .depth_image_id = scaled_depth_image,
                    .src_image_id = ssao_image0,
                    .dst_image_id = ssao_image1,
                },
            },
            &ssao_spatial_filter_task_state,
        });
        record_ctx.task_graph.add_task(SsaoUpscaleComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .g_buffer_image_id = gbuffer_depth.gbuffer,
                    .depth_image_id = gbuffer_depth.depth.task_resources.output_image,
                    .src_image_id = ssao_image1,
                    .dst_image_id = ssao_image2,
                },
            },
            &ssao_upscale_task_state,
        });
        record_ctx.task_graph.add_task(SsaoTemporalFilterComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .reprojection_image_id = reprojection_map,
                    .history_image_id = prev_ssao_image,
                    .src_image_id = ssao_image2,
                    .dst_image_id = ssao_image,
                },
            },
            &ssao_temporal_filter_task_state,
        });
        return daxa::TaskImageView{ssao_image};
    }
};

#endif

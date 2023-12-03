#pragma once

#include <shared/core.inl>

#if SSAO_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(SsaoCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, ssao_image_id)
DAXA_DECL_TASK_HEAD_END
struct SsaoComputePush {
    SsaoCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(SsaoComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewId vs_normal_image_id = push.uses.vs_normal_image_id;
daxa_ImageViewId depth_image_id = push.uses.depth_image_id;
daxa_ImageViewId ssao_image_id = push.uses.ssao_image_id;
#endif
#endif

#if SSAO_SPATIAL_FILTER_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(SsaoSpatialFilterCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, src_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, dst_image_id)
DAXA_DECL_TASK_HEAD_END
struct SsaoSpatialFilterComputePush {
    SsaoSpatialFilterCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(SsaoSpatialFilterComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId vs_normal_image_id = push.uses.vs_normal_image_id;
daxa_ImageViewId depth_image_id = push.uses.depth_image_id;
daxa_ImageViewId src_image_id = push.uses.src_image_id;
daxa_ImageViewId dst_image_id = push.uses.dst_image_id;
#endif
#endif

#if SSAO_UPSAMPLE_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(SsaoUpscaleCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, src_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, dst_image_id)
DAXA_DECL_TASK_HEAD_END
struct SsaoUpscaleComputePush {
    SsaoUpscaleCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(SsaoUpscaleComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId g_buffer_image_id = push.uses.g_buffer_image_id;
daxa_ImageViewId depth_image_id = push.uses.depth_image_id;
daxa_ImageViewId src_image_id = push.uses.src_image_id;
daxa_ImageViewId dst_image_id = push.uses.dst_image_id;
#endif
#endif

#if SSAO_TEMPORAL_FILTER_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(SsaoTemporalFilterCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, history_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, src_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, dst_image_id)
DAXA_DECL_TASK_HEAD_END
struct SsaoTemporalFilterComputePush {
    SsaoTemporalFilterCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(SsaoTemporalFilterComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId reprojection_image_id = push.uses.reprojection_image_id;
daxa_ImageViewId history_image_id = push.uses.history_image_id;
daxa_ImageViewId src_image_id = push.uses.src_image_id;
daxa_ImageViewId dst_image_id = push.uses.dst_image_id;
#endif
#endif

#if defined(__cplusplus)

struct SsaoComputeTaskState {
    AsyncManagedComputePipeline pipeline;

    SsaoComputeTaskState(AsyncPipelineManager &pipeline_manager) {
        pipeline = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"ssao.comp.glsl"},
                .compile_options = {.defines = {{"SSAO_COMPUTE", "1"}}},
            },
            .push_constant_size = sizeof(SsaoComputePush),
            .name = "ssao",
        });
    }

    void record_commands(SsaoComputePush const &push, daxa::CommandRecorder &recorder, daxa_u32vec2 render_size) {
        if (!pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(pipeline.get());
        recorder.push_constant(push);
        // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
        recorder.dispatch({(render_size.x + 7) / 8, (render_size.y + 7) / 8});
    }
};

struct SsaoSpatialFilterComputeTaskState {
    AsyncManagedComputePipeline pipeline;

    SsaoSpatialFilterComputeTaskState(AsyncPipelineManager &pipeline_manager) {
        pipeline = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"ssao.comp.glsl"},
                .compile_options = {.defines = {{"SSAO_SPATIAL_FILTER_COMPUTE", "1"}}},
            },
            .push_constant_size = sizeof(SsaoSpatialFilterComputePush),
            .name = "spatial_filter",
        });
    }

    void record_commands(SsaoSpatialFilterComputePush const &push, daxa::CommandRecorder &recorder, daxa_u32vec2 render_size) {
        if (!pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(pipeline.get());
        recorder.push_constant(push);
        // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
        recorder.dispatch({(render_size.x + 7) / 8, (render_size.y + 7) / 8});
    }
};

struct SsaoUpscaleComputeTaskState {
    AsyncManagedComputePipeline pipeline;

    SsaoUpscaleComputeTaskState(AsyncPipelineManager &pipeline_manager) {
        pipeline = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"ssao.comp.glsl"},
                .compile_options = {.defines = {{"SSAO_UPSAMPLE_COMPUTE", "1"}}},
            },
            .push_constant_size = sizeof(SsaoUpscaleComputePush),
            .name = "ssao_upscale",
        });
    }

    void record_commands(SsaoUpscaleComputePush const &push, daxa::CommandRecorder &recorder, daxa_u32vec2 render_size) {
        if (!pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(pipeline.get());
        recorder.push_constant(push);
        // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
        recorder.dispatch({(render_size.x + 7) / 8, (render_size.y + 7) / 8});
    }
};

struct SsaoTemporalFilterComputeTaskState {
    AsyncManagedComputePipeline pipeline;

    SsaoTemporalFilterComputeTaskState(AsyncPipelineManager &pipeline_manager) {
        pipeline = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"ssao.comp.glsl"},
                .compile_options = {.defines = {{"SSAO_TEMPORAL_FILTER_COMPUTE", "1"}}},
            },
            .push_constant_size = sizeof(SsaoTemporalFilterComputePush),
            .name = "ssao_temporal_filter",
        });
    }

    void record_commands(SsaoTemporalFilterComputePush const &push, daxa::CommandRecorder &recorder, daxa_u32vec2 render_size) {
        if (!pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(pipeline.get());
        recorder.push_constant(push);
        // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
        recorder.dispatch({(render_size.x + 7) / 8, (render_size.y + 7) / 8});
    }
};

struct SsaoComputeTask {
    SsaoCompute::Uses uses;
    SsaoComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto const &image_info = ti.get_device().info_image(uses.ssao_image_id.image()).value();
        auto push = SsaoComputePush{};
        ti.copy_task_head_to(&push.uses);
        state->record_commands(push, recorder, {image_info.size.x, image_info.size.y});
    }
};

struct SsaoSpatialFilterComputeTask {
    SsaoSpatialFilterCompute::Uses uses;
    SsaoSpatialFilterComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto const &image_info = ti.get_device().info_image(uses.dst_image_id.image()).value();
        auto push = SsaoSpatialFilterComputePush{};
        ti.copy_task_head_to(&push.uses);
        state->record_commands(push, recorder, {image_info.size.x, image_info.size.y});
    }
};

struct SsaoUpscaleComputeTask {
    SsaoUpscaleCompute::Uses uses;
    SsaoUpscaleComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto const &image_info = ti.get_device().info_image(uses.dst_image_id.image()).value();
        auto push = SsaoUpscaleComputePush{};
        ti.copy_task_head_to(&push.uses);
        state->record_commands(push, recorder, {image_info.size.x, image_info.size.y});
    }
};

struct SsaoTemporalFilterComputeTask {
    SsaoTemporalFilterCompute::Uses uses;
    SsaoTemporalFilterComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto const &image_info = ti.get_device().info_image(uses.dst_image_id.image()).value();
        auto push = SsaoTemporalFilterComputePush{};
        ti.copy_task_head_to(&push.uses);
        state->record_commands(push, recorder, {image_info.size.x, image_info.size.y});
    }
};

struct SsaoRenderer {
    PingPongImage ping_pong_ssao_image;
    SsaoComputeTaskState ssao_task_state;
    SsaoSpatialFilterComputeTaskState ssao_spatial_filter_task_state;
    SsaoUpscaleComputeTaskState ssao_upscale_task_state;
    SsaoTemporalFilterComputeTaskState ssao_temporal_filter_task_state;

    SsaoRenderer(AsyncPipelineManager &pipeline_manager)
        : ssao_task_state{pipeline_manager},
          ssao_spatial_filter_task_state{pipeline_manager},
          ssao_upscale_task_state{pipeline_manager},
          ssao_temporal_filter_task_state{pipeline_manager} {
    }

    void next_frame() {
        ping_pong_ssao_image.task_resources.output_resource.swap_images(ping_pong_ssao_image.task_resources.history_resource);
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
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                .vs_normal_image_id = scaled_view_normal_image,
                .depth_image_id = scaled_depth_image,
                .ssao_image_id = ssao_image0,
            },
            .state = &ssao_task_state,
        });
        record_ctx.task_graph.add_task(SsaoSpatialFilterComputeTask{
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .vs_normal_image_id = scaled_view_normal_image,
                .depth_image_id = scaled_depth_image,
                .src_image_id = ssao_image0,
                .dst_image_id = ssao_image1,
            },
            .state = &ssao_spatial_filter_task_state,
        });
        record_ctx.task_graph.add_task(SsaoUpscaleComputeTask{
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .g_buffer_image_id = gbuffer_depth.gbuffer,
                .depth_image_id = gbuffer_depth.depth.task_resources.output_resource,
                .src_image_id = ssao_image1,
                .dst_image_id = ssao_image2,
            },
            .state = &ssao_upscale_task_state,
        });
        record_ctx.task_graph.add_task(SsaoTemporalFilterComputeTask{
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .reprojection_image_id = reprojection_map,
                .history_image_id = prev_ssao_image,
                .src_image_id = ssao_image2,
                .dst_image_id = ssao_image,
            },
            .state = &ssao_temporal_filter_task_state,
        });
        return daxa::TaskImageView{ssao_image};
    }
};

#endif

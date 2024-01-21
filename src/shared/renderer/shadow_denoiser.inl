#pragma once

#include <shared/core.inl>

#if SHADOW_BIT_PACK_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(ShadowBitPackCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, output_tex)
DAXA_DECL_TASK_HEAD_END
struct ShadowBitPackComputePush {
    ShadowBitPackCompute uses;
    daxa_f32vec4 input_tex_size;
    daxa_u32vec2 bitpacked_shadow_mask_extent;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(ShadowBitPackComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId input_tex = push.uses.input_tex;
daxa_ImageViewId output_tex = push.uses.output_tex;
#endif
#endif

#if SHADOW_SPATIAL_FILTER_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(ShadowSpatialFilterCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, meta_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, geometric_normal_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, output_tex)
DAXA_DECL_TASK_HEAD_END
struct ShadowSpatialFilterComputePush {
    ShadowSpatialFilterCompute uses;
    daxa_f32vec4 input_tex_size;
    daxa_u32vec2 bitpacked_shadow_mask_extent;
    daxa_u32 step_size;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(ShadowSpatialFilterComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewId input_tex = push.uses.input_tex;
daxa_ImageViewId meta_tex = push.uses.meta_tex;
daxa_ImageViewId geometric_normal_tex = push.uses.geometric_normal_tex;
daxa_ImageViewId depth_tex = push.uses.depth_tex;
daxa_ImageViewId output_tex = push.uses.output_tex;
#endif
#endif

#if SHADOW_TEMPORAL_FILTER_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(ShadowTemporalFilterCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, shadow_mask_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, bitpacked_shadow_mask_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, prev_moments_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, prev_accum_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, output_moments_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, temporal_output_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, meta_output_tex)
DAXA_DECL_TASK_HEAD_END
struct ShadowTemporalFilterComputePush {
    ShadowTemporalFilterCompute uses;
    daxa_f32vec4 input_tex_size;
    daxa_u32vec2 bitpacked_shadow_mask_extent;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(ShadowTemporalFilterComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewId shadow_mask_tex = push.uses.shadow_mask_tex;
daxa_ImageViewId bitpacked_shadow_mask_tex = push.uses.bitpacked_shadow_mask_tex;
daxa_ImageViewId prev_moments_tex = push.uses.prev_moments_tex;
daxa_ImageViewId prev_accum_tex = push.uses.prev_accum_tex;
daxa_ImageViewId reprojection_tex = push.uses.reprojection_tex;
daxa_ImageViewId output_moments_tex = push.uses.output_moments_tex;
daxa_ImageViewId temporal_output_tex = push.uses.temporal_output_tex;
daxa_ImageViewId meta_output_tex = push.uses.meta_output_tex;
#endif
#endif

#if defined(__cplusplus)

struct ShadowBitPackComputeTaskState {
    AsyncManagedComputePipeline pipeline;
    ShadowBitPackComputeTaskState(AsyncPipelineManager &pipeline_manager) {
        pipeline = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"shadow_denoiser.comp.glsl"},
                .compile_options = {.defines = {{"SHADOW_BIT_PACK_COMPUTE", "1"}}},
            },
            .push_constant_size = sizeof(ShadowBitPackComputePush),
            .name = "shadow_bit_pack",
        });
    }
};

struct ShadowSpatialFilterComputeTaskState {
    AsyncManagedComputePipeline pipeline;
    ShadowSpatialFilterComputeTaskState(AsyncPipelineManager &pipeline_manager) {
        pipeline = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"shadow_denoiser.comp.glsl"},
                .compile_options = {.defines = {{"SHADOW_SPATIAL_FILTER_COMPUTE", "1"}}},
            },
            .push_constant_size = sizeof(ShadowSpatialFilterComputePush),
            .name = "shadow_spatial_filter",
        });
    }
};

struct ShadowTemporalFilterComputeTaskState {
    AsyncManagedComputePipeline pipeline;
    ShadowTemporalFilterComputeTaskState(AsyncPipelineManager &pipeline_manager) {
        pipeline = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"shadow_denoiser.comp.glsl"},
                .compile_options = {.defines = {{"SHADOW_TEMPORAL_FILTER_COMPUTE", "1"}}},
            },
            .push_constant_size = sizeof(ShadowTemporalFilterComputePush),
            .name = "shadow_temporal_filter",
        });
    }
};

struct ShadowBitPackComputeTask {
    ShadowBitPackCompute::Uses uses;
    std::string name = "ShadowBitPackCompute";
    ShadowBitPackComputeTaskState *state;
    daxa_u32 step_size;
    daxa_u32vec2 bitpacked_shadow_mask_extent;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto const &image_info = ti.get_device().info_image(uses.input_tex.image()).value();
        auto push = ShadowBitPackComputePush{};
        ti.copy_task_head_to(&push.uses);
        push.input_tex_size = daxa_f32vec4{float(image_info.size.x), float(image_info.size.y), 0.0f, 0.0f};
        push.input_tex_size.z = 1.0f / push.input_tex_size.x;
        push.input_tex_size.w = 1.0f / push.input_tex_size.y;
        push.bitpacked_shadow_mask_extent = bitpacked_shadow_mask_extent;
        auto render_size = daxa_u32vec2{bitpacked_shadow_mask_extent.x * 2, bitpacked_shadow_mask_extent.y};
        if (!state->pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(state->pipeline.get());
        recorder.push_constant(push);
        // assert((render_size.x % 8) == 0 && (render_size.y % 4) == 0);
        recorder.dispatch({(render_size.x + 7) / 8, (render_size.y + 3) / 4});
    }
};

struct ShadowSpatialFilterComputeTask {
    ShadowSpatialFilterCompute::Uses uses;
    std::string name = "ShadowSpatialFilterCompute";
    ShadowSpatialFilterComputeTaskState *state;
    daxa_u32 step_size;
    daxa_u32vec2 bitpacked_shadow_mask_extent;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto const &image_info = ti.get_device().info_image(uses.output_tex.image()).value();
        auto const &input_image_info = ti.get_device().info_image(uses.input_tex.image()).value();
        auto push = ShadowSpatialFilterComputePush{};
        ti.copy_task_head_to(&push.uses);
        push.step_size = step_size;
        push.input_tex_size = daxa_f32vec4{float(input_image_info.size.x), float(input_image_info.size.y), 0.0f, 0.0f};
        push.input_tex_size.z = 1.0f / push.input_tex_size.x;
        push.input_tex_size.w = 1.0f / push.input_tex_size.y;
        push.bitpacked_shadow_mask_extent = bitpacked_shadow_mask_extent;
        if (!state->pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(state->pipeline.get());
        recorder.push_constant(push);
        // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
        recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
    }
};

struct ShadowTemporalFilterComputeTask {
    ShadowTemporalFilterCompute::Uses uses;
    std::string name = "ShadowTemporalFilterCompute";
    ShadowTemporalFilterComputeTaskState *state;
    daxa_u32vec2 bitpacked_shadow_mask_extent;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto const &image_info = ti.get_device().info_image(uses.output_moments_tex.image()).value();
        auto const &input_image_info = ti.get_device().info_image(uses.shadow_mask_tex.image()).value();
        auto push = ShadowTemporalFilterComputePush{};
        ti.copy_task_head_to(&push.uses);
        push.input_tex_size = daxa_f32vec4{float(input_image_info.size.x), float(input_image_info.size.y), 0.0f, 0.0f};
        push.input_tex_size.z = 1.0f / push.input_tex_size.x;
        push.input_tex_size.w = 1.0f / push.input_tex_size.y;
        push.bitpacked_shadow_mask_extent = bitpacked_shadow_mask_extent;
        if (!state->pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(state->pipeline.get());
        recorder.push_constant(push);
        // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
        recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 3) / 4});
    }
};

struct ShadowDenoiser {
    ShadowBitPackComputeTaskState shadow_bit_pack_task_state;
    ShadowTemporalFilterComputeTaskState shadow_temporal_filter_task_state;
    ShadowSpatialFilterComputeTaskState shadow_spatial_filter_task_state;
    PingPongImage ping_pong_moments_image;
    PingPongImage ping_pong_accum_image;

    ShadowDenoiser(AsyncPipelineManager &pipeline_manager)
        : shadow_bit_pack_task_state{pipeline_manager},
          shadow_temporal_filter_task_state{pipeline_manager},
          shadow_spatial_filter_task_state{pipeline_manager} {
    }

    void next_frame() {
        ping_pong_moments_image.task_resources.output_resource.swap_images(ping_pong_moments_image.task_resources.history_resource);
        ping_pong_accum_image.task_resources.output_resource.swap_images(ping_pong_accum_image.task_resources.history_resource);
    }

    auto denoise_shadow_bitmap(RecordContext &record_ctx, GbufferDepth const &gbuffer_depth, daxa::TaskImageView shadow_bitmap, daxa::TaskImageView reprojection_map) -> daxa::TaskImageView {
        auto bitpacked_shadow_mask_extent = daxa_u32vec2{(record_ctx.render_resolution.x + 7) / 8, (record_ctx.render_resolution.y + 3) / 4};

        auto bitpacked_shadows_image = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R32_UINT,
            .size = {bitpacked_shadow_mask_extent.x, bitpacked_shadow_mask_extent.y, 1},
            .name = "bitpacked_shadows_image",
        });
        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "shadow bitpack", .task_image_id = bitpacked_shadows_image, .type = DEBUG_IMAGE_TYPE_SHADOW_BITMAP});

        record_ctx.task_graph.add_task(ShadowBitPackComputeTask{
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .input_tex = shadow_bitmap,
                .output_tex = bitpacked_shadows_image,
            },
            .state = &shadow_bit_pack_task_state,
            .bitpacked_shadow_mask_extent = bitpacked_shadow_mask_extent,
        });

        ping_pong_moments_image = PingPongImage{};
        auto [moments_image, prev_moments_image] = ping_pong_moments_image.get(
            record_ctx.device,
            {
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC,
                .name = "moments_image",
            });
        record_ctx.task_graph.use_persistent_image(moments_image);
        record_ctx.task_graph.use_persistent_image(prev_moments_image);

        ping_pong_accum_image = PingPongImage{};
        auto [accum_image, prev_accum_image] = ping_pong_accum_image.get(
            record_ctx.device,
            {
                .format = daxa::Format::R16G16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC,
                .name = "accum_image",
            });
        record_ctx.task_graph.use_persistent_image(accum_image);
        record_ctx.task_graph.use_persistent_image(prev_accum_image);

        auto spatial_input_image = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "spatial_input_image",
        });
        auto shadow_denoise_intermediary_1 = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "shadow_denoise_intermediary_1",
        });

        auto metadata_image = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R32_UINT,
            .size = {bitpacked_shadow_mask_extent.x, bitpacked_shadow_mask_extent.y, 1},
            .name = "metadata_image",
        });

        record_ctx.task_graph.add_task(ShadowTemporalFilterComputeTask{
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                .shadow_mask_tex = shadow_bitmap,
                .bitpacked_shadow_mask_tex = bitpacked_shadows_image,
                .prev_moments_tex = prev_moments_image,
                .prev_accum_tex = prev_accum_image,
                .reprojection_tex = reprojection_map,
                .output_moments_tex = moments_image,
                .temporal_output_tex = spatial_input_image,
                .meta_output_tex = metadata_image,
            },
            .state = &shadow_temporal_filter_task_state,
            .bitpacked_shadow_mask_extent = bitpacked_shadow_mask_extent,
        });

        record_ctx.task_graph.add_task(ShadowSpatialFilterComputeTask{
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                .input_tex = spatial_input_image,
                .meta_tex = metadata_image,
                .geometric_normal_tex = gbuffer_depth.geometric_normal,
                .depth_tex = gbuffer_depth.depth.task_resources.output_resource,
                .output_tex = accum_image,
            },
            .state = &shadow_spatial_filter_task_state,
            .step_size = 1,
            .bitpacked_shadow_mask_extent = bitpacked_shadow_mask_extent,
        });

        record_ctx.task_graph.add_task(ShadowSpatialFilterComputeTask{
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                .input_tex = accum_image,
                .meta_tex = metadata_image,
                .geometric_normal_tex = gbuffer_depth.geometric_normal,
                .depth_tex = gbuffer_depth.depth.task_resources.output_resource,
                .output_tex = shadow_denoise_intermediary_1,
            },
            .state = &shadow_spatial_filter_task_state,
            .step_size = 2,
            .bitpacked_shadow_mask_extent = bitpacked_shadow_mask_extent,
        });

        record_ctx.task_graph.add_task(ShadowSpatialFilterComputeTask{
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                .input_tex = shadow_denoise_intermediary_1,
                .meta_tex = metadata_image,
                .geometric_normal_tex = gbuffer_depth.geometric_normal,
                .depth_tex = gbuffer_depth.depth.task_resources.output_resource,
                .output_tex = spatial_input_image,
            },
            .state = &shadow_spatial_filter_task_state,
            .step_size = 4,
            .bitpacked_shadow_mask_extent = bitpacked_shadow_mask_extent,
        });

        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "shadow temporal", .task_image_id = moments_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "shadow spatial0", .task_image_id = accum_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "shadow spatial1", .task_image_id = shadow_denoise_intermediary_1, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "shadow spatial2", .task_image_id = spatial_input_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        return spatial_input_image;
    }
};

#endif

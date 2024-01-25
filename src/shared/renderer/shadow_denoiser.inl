#pragma once

#include <shared/core.inl>

#if ShadowBitPackComputeShader || defined(__cplusplus)
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

#if ShadowSpatialFilterComputeShader || defined(__cplusplus)
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

#if ShadowTemporalFilterComputeShader || defined(__cplusplus)
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

struct ShadowDenoiser {
    PingPongImage ping_pong_moments_image;
    PingPongImage ping_pong_accum_image;

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

        struct ShadowBitPackComputeInfo {
            daxa_u32vec2 bitpacked_shadow_mask_extent;
        };
        record_ctx.add(ComputeTask<ShadowBitPackCompute, ShadowBitPackComputePush, ShadowBitPackComputeInfo>{
            .source = daxa::ShaderFile{"shadow_denoiser.comp.glsl"},
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .input_tex = shadow_bitmap,
                .output_tex = bitpacked_shadows_image,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ShadowBitPackCompute::Uses &uses, ShadowBitPackComputePush &push, ShadowBitPackComputeInfo const &info) {
                auto const &image_info = ti.get_device().info_image(uses.input_tex.image()).value();
                push.input_tex_size = daxa_f32vec4{float(image_info.size.x), float(image_info.size.y), 0.0f, 0.0f};
                push.input_tex_size.z = 1.0f / push.input_tex_size.x;
                push.input_tex_size.w = 1.0f / push.input_tex_size.y;
                push.bitpacked_shadow_mask_extent = info.bitpacked_shadow_mask_extent;
                auto render_size = daxa_u32vec2{info.bitpacked_shadow_mask_extent.x * 2, info.bitpacked_shadow_mask_extent.y};
                ti.get_recorder().set_pipeline(pipeline);
                ti.get_recorder().push_constant(push);
                // assert((render_size.x % 8) == 0 && (render_size.y % 4) == 0);
                ti.get_recorder().dispatch({(render_size.x + 7) / 8, (render_size.y + 3) / 4});
            },
            .info = {
                .bitpacked_shadow_mask_extent = bitpacked_shadow_mask_extent,
            },
        });

        ping_pong_moments_image = PingPongImage{};
        auto [moments_image, prev_moments_image] = ping_pong_moments_image.get(
            record_ctx.device,
            {
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
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
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "accum_image",
            });
        record_ctx.task_graph.use_persistent_image(accum_image);
        record_ctx.task_graph.use_persistent_image(prev_accum_image);

        clear_task_images(record_ctx.device, std::array{prev_moments_image, prev_accum_image});

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

        record_ctx.add(ComputeTask<ShadowTemporalFilterCompute, ShadowTemporalFilterComputePush, ShadowBitPackComputeInfo>{
            .source = daxa::ShaderFile{"shadow_denoiser.comp.glsl"},
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
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ShadowTemporalFilterCompute::Uses &uses, ShadowTemporalFilterComputePush &push, ShadowBitPackComputeInfo const &info) {
                auto const &image_info = ti.get_device().info_image(uses.output_moments_tex.image()).value();
                auto const &input_image_info = ti.get_device().info_image(uses.shadow_mask_tex.image()).value();
                push.input_tex_size = daxa_f32vec4{float(input_image_info.size.x), float(input_image_info.size.y), 0.0f, 0.0f};
                push.input_tex_size.z = 1.0f / push.input_tex_size.x;
                push.input_tex_size.w = 1.0f / push.input_tex_size.y;
                push.bitpacked_shadow_mask_extent = info.bitpacked_shadow_mask_extent;
                ti.get_recorder().set_pipeline(pipeline);
                ti.get_recorder().push_constant(push);
                // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
                ti.get_recorder().dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 3) / 4});
            },
            .info = {
                .bitpacked_shadow_mask_extent = bitpacked_shadow_mask_extent,
            },
        });

        struct ShadowSpatialFilterComputeInfo {
            daxa_u32 step_size;
            daxa_u32vec2 bitpacked_shadow_mask_extent;
        };
        auto shadow_spatial_task_callback = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ShadowSpatialFilterCompute::Uses &uses, ShadowSpatialFilterComputePush &push, ShadowSpatialFilterComputeInfo const &info) {
            auto const &image_info = ti.get_device().info_image(uses.output_tex.image()).value();
            auto const &input_image_info = ti.get_device().info_image(uses.input_tex.image()).value();
            push.step_size = info.step_size;
            push.input_tex_size = daxa_f32vec4{float(input_image_info.size.x), float(input_image_info.size.y), 0.0f, 0.0f};
            push.input_tex_size.z = 1.0f / push.input_tex_size.x;
            push.input_tex_size.w = 1.0f / push.input_tex_size.y;
            push.bitpacked_shadow_mask_extent = info.bitpacked_shadow_mask_extent;
            ti.get_recorder().set_pipeline(pipeline);
            ti.get_recorder().push_constant(push);
            // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
            ti.get_recorder().dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
        };

        record_ctx.add(ComputeTask<ShadowSpatialFilterCompute, ShadowSpatialFilterComputePush, ShadowSpatialFilterComputeInfo>{
            .source = daxa::ShaderFile{"shadow_denoiser.comp.glsl"},
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                .input_tex = spatial_input_image,
                .meta_tex = metadata_image,
                .geometric_normal_tex = gbuffer_depth.geometric_normal,
                .depth_tex = gbuffer_depth.depth.task_resources.output_resource,
                .output_tex = accum_image,
            },
            .callback_ = shadow_spatial_task_callback,
            .info = {
                .step_size = 1,
                .bitpacked_shadow_mask_extent = bitpacked_shadow_mask_extent,
            },
        });

        record_ctx.add(ComputeTask<ShadowSpatialFilterCompute, ShadowSpatialFilterComputePush, ShadowSpatialFilterComputeInfo>{
            .source = daxa::ShaderFile{"shadow_denoiser.comp.glsl"},
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                .input_tex = accum_image,
                .meta_tex = metadata_image,
                .geometric_normal_tex = gbuffer_depth.geometric_normal,
                .depth_tex = gbuffer_depth.depth.task_resources.output_resource,
                .output_tex = shadow_denoise_intermediary_1,
            },
            .callback_ = shadow_spatial_task_callback,
            .info = {
                .step_size = 2,
                .bitpacked_shadow_mask_extent = bitpacked_shadow_mask_extent,
            },
        });

        record_ctx.add(ComputeTask<ShadowSpatialFilterCompute, ShadowSpatialFilterComputePush, ShadowSpatialFilterComputeInfo>{
            .source = daxa::ShaderFile{"shadow_denoiser.comp.glsl"},
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                .input_tex = shadow_denoise_intermediary_1,
                .meta_tex = metadata_image,
                .geometric_normal_tex = gbuffer_depth.geometric_normal,
                .depth_tex = gbuffer_depth.depth.task_resources.output_resource,
                .output_tex = spatial_input_image,
            },
            .callback_ = shadow_spatial_task_callback,
            .info = {
                .step_size = 4,
                .bitpacked_shadow_mask_extent = bitpacked_shadow_mask_extent,
            },
        });

        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "shadow temporal", .task_image_id = moments_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "shadow spatial0", .task_image_id = accum_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "shadow spatial1", .task_image_id = shadow_denoise_intermediary_1, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "shadow spatial2", .task_image_id = spatial_input_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        return spatial_input_image;
    }
};

#endif

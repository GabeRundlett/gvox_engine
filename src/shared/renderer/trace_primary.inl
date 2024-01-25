#pragma once

#include <shared/core.inl>

#if TraceDepthPrepassComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(TraceDepthPrepassCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, render_depth_prepass_image)
DAXA_DECL_TASK_HEAD_END
struct TraceDepthPrepassComputePush {
    TraceDepthPrepassCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(TraceDepthPrepassComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewId render_depth_prepass_image = push.uses.render_depth_prepass_image;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)
#endif
#endif

#if TracePrimaryComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(TracePrimaryCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, sky_lut)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_3D, blue_noise_vec2)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, debug_texture)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, render_depth_prepass_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct TracePrimaryComputePush {
    TracePrimaryCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(TracePrimaryComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)
daxa_ImageViewId sky_lut = push.uses.sky_lut;
daxa_ImageViewId blue_noise_vec2 = push.uses.blue_noise_vec2;
daxa_ImageViewId debug_texture = push.uses.debug_texture;
daxa_ImageViewId render_depth_prepass_image = push.uses.render_depth_prepass_image;
daxa_ImageViewId g_buffer_image_id = push.uses.g_buffer_image_id;
daxa_ImageViewId vs_normal_image_id = push.uses.vs_normal_image_id;
daxa_ImageViewId velocity_image_id = push.uses.velocity_image_id;
daxa_ImageViewId depth_image_id = push.uses.depth_image_id;
#endif
#endif

#if defined(__cplusplus)

struct GbufferRenderer {
    GbufferDepth gbuffer_depth;

    void next_frame() {
        gbuffer_depth.next_frame();
    }

    auto render(RecordContext &record_ctx, daxa::TaskImageView sky_lut, VoxelWorld::Buffers &voxel_buffers)
        -> std::pair<GbufferDepth &, daxa::TaskImageView> {
        gbuffer_depth.gbuffer = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R32G32B32A32_UINT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "gbuffer",
        });
        gbuffer_depth.geometric_normal = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::A2B10G10R10_UNORM_PACK32,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "normal",
        });

        gbuffer_depth.downscaled_view_normal = std::nullopt;
        gbuffer_depth.downscaled_depth = std::nullopt;

        gbuffer_depth.depth = PingPongImage{};
        auto [depth_image, prev_depth_image] = gbuffer_depth.depth.get(
            record_ctx.device,
            {
                .format = daxa::Format::R32_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC,
                .name = "depth_image",
            });

        record_ctx.task_graph.use_persistent_image(depth_image);
        record_ctx.task_graph.use_persistent_image(prev_depth_image);

        auto velocity_image = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "velocity_image",
        });

        auto depth_prepass_image = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R32_SFLOAT,
            .size = {record_ctx.render_resolution.x / PREPASS_SCL, record_ctx.render_resolution.y / PREPASS_SCL, 1},
            .name = "depth_prepass_image",
        });

        record_ctx.add(ComputeTask<TraceDepthPrepassCompute, TraceDepthPrepassComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"trace_primary.comp.glsl"},
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                VOXELS_BUFFER_USES_ASSIGN(voxel_buffers),
                .render_depth_prepass_image = depth_prepass_image,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TraceDepthPrepassCompute::Uses &uses, TraceDepthPrepassComputePush &push, NoTaskInfo const &) {
                auto const &image_info = ti.get_device().info_image(uses.render_depth_prepass_image.image()).value();
                ti.get_recorder().set_pipeline(pipeline);
                ti.get_recorder().push_constant(push);
                // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
                ti.get_recorder().dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });

        record_ctx.add(ComputeTask<TracePrimaryCompute, TracePrimaryComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"trace_primary.comp.glsl"},
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                VOXELS_BUFFER_USES_ASSIGN(voxel_buffers),
                .sky_lut = sky_lut,
                .blue_noise_vec2 = record_ctx.task_blue_noise_vec2_image,
                .debug_texture = record_ctx.task_debug_texture,
                .render_depth_prepass_image = depth_prepass_image,
                .g_buffer_image_id = gbuffer_depth.gbuffer,
                .vs_normal_image_id = gbuffer_depth.geometric_normal,
                .velocity_image_id = velocity_image,
                .depth_image_id = depth_image,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TracePrimaryCompute::Uses &uses, TracePrimaryComputePush &push, NoTaskInfo const &) {
                auto const &image_info = ti.get_device().info_image(uses.g_buffer_image_id.image()).value();
                ti.get_recorder().set_pipeline(pipeline);
                ti.get_recorder().push_constant(push);
                // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
                ti.get_recorder().dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });

        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "depth_prepass", .task_image_id = depth_prepass_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "gbuffer", .task_image_id = gbuffer_depth.gbuffer, .type = DEBUG_IMAGE_TYPE_GBUFFER});
        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "geometric_normal", .task_image_id = gbuffer_depth.geometric_normal, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "velocity", .task_image_id = velocity_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        return {gbuffer_depth, velocity_image};
    }
};

#endif

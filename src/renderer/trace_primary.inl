#pragma once

#include <core.inl>
#include <renderer/core.inl>

// #include <voxels/particles/voxel_particles.inl>

DAXA_DECL_TASK_HEAD_BEGIN(R32D32Blit)
DAXA_TH_IMAGE_INDEX(FRAGMENT_SHADER_SAMPLED, REGULAR_2D, input_tex)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, output_tex)
DAXA_DECL_TASK_HEAD_END
struct R32D32BlitPush {
    DAXA_TH_BLOB(R32D32Blit, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(TracePrimaryRt)
DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
// DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(daxa_BufferPtr(BlasGeom)), geometry_pointers)
// DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(daxa_BufferPtr(VoxelBrickAttribs)), attribute_pointers)
// DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(VoxelBlasTransform), blas_transforms)
DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(daxa_BufferPtr(ChunkPrimitive)), chunk_primitive_pointers)
DAXA_TH_TLAS_PTR(RAY_TRACING_SHADER_READ, tlas)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
#if defined(DAXA_RAY_TRACING) || defined(__cplusplus)
struct TracePrimaryRtPush {
    DAXA_TH_BLOB(TracePrimaryRt, uses)
};
#endif

#if defined(__cplusplus)

#include <application/settings.hpp>

struct GbufferRenderer {
    GbufferDepth gbuffer_depth;

    void next_frame() {
        gbuffer_depth.next_frame();
    }

    auto render(GpuContext &gpu_context, VoxelWorldBuffers &voxel_buffers)
        -> std::pair<GbufferDepth &, daxa::TaskImageView> {
        gbuffer_depth.gbuffer = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R32G32B32A32_UINT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "gbuffer",
        });
        gbuffer_depth.geometric_normal = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::A2B10G10R10_UNORM_PACK32,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "normal",
        });

        gbuffer_depth.downscaled_view_normal = std::nullopt;
        gbuffer_depth.downscaled_depth = std::nullopt;

        gbuffer_depth.depth = PingPongImage{};
        auto [depth_image, prev_depth_image] = gbuffer_depth.depth.get(
            gpu_context,
            {
                .format = daxa::Format::D32_SFLOAT,
                .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT,
                .name = "depth_image",
            });

        gpu_context.frame_task_graph.use_persistent_image(depth_image);
        gpu_context.frame_task_graph.use_persistent_image(prev_depth_image);

        auto velocity_image = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "velocity_image",
        });

        auto temp_depth_image = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R32_SFLOAT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "temp_depth_image",
        });

        gpu_context.add(RayTracingTask<TracePrimaryRt::Task, TracePrimaryRtPush, NoTaskInfo>{
            .source = daxa::ShaderFile{"trace_primary.rt.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TracePrimaryRt::AT.gpu_input, gpu_context.task_input_buffer}},
                // daxa::TaskViewVariant{std::pair{TracePrimaryRt::AT.chunk_primitive_pointers, voxel_buffers.chunk_primitive_pointers.task_resource}},
                // daxa::TaskViewVariant{std::pair{TracePrimaryRt::AT.attribute_pointers, voxel_buffers.blas_attr_pointers.task_resource}},
                // daxa::TaskViewVariant{std::pair{TracePrimaryRt::AT.blas_transforms, voxel_buffers.blas_transforms.task_resource}},
                daxa::TaskViewVariant{std::pair{TracePrimaryRt::AT.chunk_primitive_pointers, voxel_buffers.brick_primitive_pointers.task_resource}},
                daxa::TaskViewVariant{std::pair{TracePrimaryRt::AT.tlas, voxel_buffers.task_tlas}},
                daxa::TaskViewVariant{std::pair{TracePrimaryRt::AT.g_buffer_image_id, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{TracePrimaryRt::AT.velocity_image_id, velocity_image}},
                daxa::TaskViewVariant{std::pair{TracePrimaryRt::AT.vs_normal_image_id, gbuffer_depth.geometric_normal}},
                daxa::TaskViewVariant{std::pair{TracePrimaryRt::AT.depth_image_id, temp_depth_image}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RayTracingPipeline &pipeline, TracePrimaryRtPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(TracePrimaryRt::AT.g_buffer_image_id).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.trace_rays({.width = image_info.size.x, .height = image_info.size.y, .depth = 1});
            },
        });

        gpu_context.add(RasterTask<R32D32Blit::Task, R32D32BlitPush, NoTaskInfo>{
            .vert_source = daxa::ShaderFile{"FULL_SCREEN_TRIANGLE_VERTEX_SHADER"},
            .frag_source = daxa::ShaderFile{"R32_D32_BLIT"},
            .depth_test = daxa::DepthTestInfo{
                .depth_attachment_format = daxa::Format::D32_SFLOAT,
                .enable_depth_write = true,
                .depth_test_compare_op = daxa::CompareOp::ALWAYS,
            },
            .views = std::array{
                daxa::TaskViewVariant{std::pair{R32D32Blit::AT.input_tex, temp_depth_image}},
                daxa::TaskViewVariant{std::pair{R32D32Blit::AT.output_tex, depth_image}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, R32D32BlitPush &push, NoTaskInfo const &) {
                auto render_image = ti.get(R32D32Blit::AT.output_tex).ids[0];
                auto const image_info = ti.device.info_image(render_image).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .depth_attachment = {{.image_view = ti.get(R32D32Blit::AT.output_tex).view_ids[0], .load_op = daxa::AttachmentLoadOp::CLEAR, .clear_value = std::array{0.0f, 0.0f, 0.0f, 0.0f}}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.draw({.vertex_count = 3});
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "gbuffer", .task_image_id = gbuffer_depth.gbuffer, .type = DEBUG_IMAGE_TYPE_GBUFFER});
        debug_utils::DebugDisplay::add_pass({.name = "temp_depth_image", .task_image_id = temp_depth_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "depth", .task_image_id = depth_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "geometric_normal", .task_image_id = gbuffer_depth.geometric_normal, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "velocity", .task_image_id = velocity_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        return {gbuffer_depth, velocity_image};
    }
};

#endif

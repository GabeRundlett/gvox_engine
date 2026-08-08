#pragma once

#include <core.inl>
#include <renderer/core.inl>

// #include <voxels/particles/voxel_particles.inl>

DAXA_DECL_RAY_TRACING_TASK_HEAD_BEGIN(TracePrimaryRt)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
// DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_BufferPtr(BlasGeom)), geometry_pointers)
// DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_BufferPtr(VoxelBrickAttribs)), attribute_pointers)
// DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VoxelBlasTransform), blas_transforms)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuVoxelObject), voxel_object_manifests)
DAXA_TH_TLAS_PTR(READ, tlas)
DAXA_TH_IMAGE_INDEX(WRITE, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE_INDEX(WRITE, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE_INDEX(WRITE, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE_INDEX(WRITE, REGULAR_2D, depth_image_id)
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
        gbuffer_depth.gbuffer = gpu_context.frame_task_graph.create_task_image({
            .format = daxa::Format::R32G32B32A32_UINT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "gbuffer",
        });
        gbuffer_depth.geometric_normal = gpu_context.frame_task_graph.create_task_image({
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

        gpu_context.frame_task_graph.register_image(depth_image);
        gpu_context.frame_task_graph.register_image(prev_depth_image);

        auto velocity_image = gpu_context.frame_task_graph.create_task_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "velocity_image",
        });

        auto temp_depth_image = gpu_context.frame_task_graph.create_task_image({
            .format = daxa::Format::R32_SFLOAT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "temp_depth_image",
        });

        gpu_context.add(RayTracingTask<TracePrimaryRt::Info, TracePrimaryRtPush, NoTaskInfo>{
            .source = daxa::ShaderFile{"trace_primary.rt.glsl"},
            .views = TracePrimaryRt::Views{
                .gpu_input = gpu_context.task_input_buffer.view(),
                // .voxel_object_manifests = voxel_buffers.voxel_object_manifests.task_resource.view(),
                // .attribute_pointers = voxel_buffers.blas_attr_pointers.task_resource.view(),
                // .blas_transforms = voxel_buffers.blas_transforms.task_resource.view(),
                .voxel_object_manifests = voxel_buffers.voxel_object_manifests.task_resource.view(),
                .tlas = voxel_buffers.task_tlas.view(),
                .g_buffer_image_id = gbuffer_depth.gbuffer,
                .velocity_image_id = velocity_image,
                .vs_normal_image_id = gbuffer_depth.geometric_normal,
                .depth_image_id = temp_depth_image,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RayTracingPipeline &pipeline, daxa::RayTracingShaderBindingTable const &shader_binding_table, TracePrimaryRtPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.image_info(ti.get(TracePrimaryRt::AT.g_buffer_image_id).id).value();
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.trace_rays({.width = image_info.size.x, .height = image_info.size.y, .depth = 1, .shader_binding_table = shader_binding_table});
            },
        });

        r32_d32_blit(gpu_context, temp_depth_image, depth_image.view());

        debug_utils::DebugDisplay::add_pass({.name = "gbuffer", .task_image_id = gbuffer_depth.gbuffer, .type = DEBUG_IMAGE_TYPE_GBUFFER});
        debug_utils::DebugDisplay::add_pass({.name = "temp_depth_image", .task_image_id = temp_depth_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "depth", .task_image_id = depth_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "geometric_normal", .task_image_id = gbuffer_depth.geometric_normal, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "velocity", .task_image_id = velocity_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        return {gbuffer_depth, velocity_image};
    }
};

#endif

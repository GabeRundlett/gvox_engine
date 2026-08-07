#pragma once

#include <core.inl>
#include <renderer/core.inl>

DAXA_DECL_TASK_HEAD_BEGIN(TraceShadowRt)
DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
// DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(daxa_BufferPtr(BlasGeom)), geometry_pointers)
// DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(daxa_BufferPtr(VoxelBrickAttribs)), attribute_pointers)
// DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(VoxelBlasTransform), blas_transforms)
DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(daxa_BufferPtr(ChunkPrimitive)), chunk_primitive_pointers)
DAXA_TH_TLAS_PTR(RAY_TRACING_SHADER_READ, tlas)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_SAMPLED, REGULAR_3D, blue_noise_vec2)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_SAMPLED, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_SAMPLED, REGULAR_2D, depth_image_id)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_SAMPLED, REGULAR_2D, particles_shadow_depth_tex)
// TODO: Figure out why this needs to be an ID...
DAXA_TH_IMAGE_ID(RAY_TRACING_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, shadow_mask)
DAXA_DECL_TASK_HEAD_END
#if defined(DAXA_RAY_TRACING) || defined(__cplusplus)
struct TraceShadowRtPush {
    DAXA_TH_BLOB(TraceShadowRt, uses)
};
#endif

#if defined(__cplusplus)

#include <application/settings.hpp>

inline auto trace_shadows(GpuContext &gpu_context, GbufferDepth &gbuffer_depth, VoxelWorldBuffers &voxel_buffers, daxa::TaskImageView particles_shadow_depth_image) -> daxa::TaskImageView {
    auto shadow_mask = gpu_context.frame_task_graph.create_transient_image({
        .format = daxa::Format::R8_UNORM,
        .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
        .name = "shadow_mask",
    });

    AppSettings::add<settings::Checkbox>({"Graphics", "Render Shadows", {.value = true}, {.task_graph_depends = true}});

    auto render_shadows = AppSettings::get<settings::Checkbox>("Graphics", "Render Shadows").value;

    if (render_shadows) {
        gpu_context.add(RayTracingTask<TraceShadowRt::Task, TraceShadowRtPush, NoTaskInfo>{
            .source = daxa::ShaderFile{"trace_shadow.rt.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TraceShadowRt::AT.gpu_input, gpu_context.task_input_buffer}},
                // daxa::TaskViewVariant{std::pair{TraceShadowRt::AT.geometry_pointers, voxel_buffers.blas_geom_pointers.task_resource}},
                // daxa::TaskViewVariant{std::pair{TraceShadowRt::AT.attribute_pointers, voxel_buffers.blas_attr_pointers.task_resource}},
                // daxa::TaskViewVariant{std::pair{TraceShadowRt::AT.blas_transforms, voxel_buffers.blas_transforms.task_resource}},
                daxa::TaskViewVariant{std::pair{TraceShadowRt::AT.chunk_primitive_pointers, voxel_buffers.brick_primitive_pointers.task_resource}},
                daxa::TaskViewVariant{std::pair{TraceShadowRt::AT.tlas, voxel_buffers.task_tlas}},
                daxa::TaskViewVariant{std::pair{TraceShadowRt::AT.blue_noise_vec2, gpu_context.task_blue_noise_vec2_image}},
                daxa::TaskViewVariant{std::pair{TraceShadowRt::AT.g_buffer_image_id, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{TraceShadowRt::AT.depth_image_id, gbuffer_depth.depth.current()}},
                daxa::TaskViewVariant{std::pair{TraceShadowRt::AT.particles_shadow_depth_tex, particles_shadow_depth_image}},
                daxa::TaskViewVariant{std::pair{TraceShadowRt::AT.shadow_mask, shadow_mask}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RayTracingPipeline &pipeline, TraceShadowRtPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(TraceShadowRt::AT.g_buffer_image_id).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.trace_rays({.width = image_info.size.x, .height = image_info.size.y, .depth = 1});
            },
        });
    } else {
        clear_task_images(gpu_context.frame_task_graph, std::array<daxa::TaskImageView, 1>{shadow_mask}, std::array<daxa::ClearValue, 1>{std::array<float, 4>{1.0f, 1.0f, 1.0f, 1.0f}});
    }

    debug_utils::DebugDisplay::add_pass({.name = "trace shadow bitmap", .task_image_id = shadow_mask, .type = DEBUG_IMAGE_TYPE_DEFAULT_UINT});

    return daxa::TaskImageView{shadow_mask};
}

#endif

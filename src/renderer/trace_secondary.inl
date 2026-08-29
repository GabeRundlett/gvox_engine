#ifndef RENDERER_TRACE_SECONDARY_INL
#define RENDERER_TRACE_SECONDARY_INL

#include <core.inl>
#include <renderer/core.inl>

#if USE_RAY_QUERY
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(TraceShadowRt)
#else
DAXA_DECL_RAY_TRACING_TASK_HEAD_BEGIN(TraceShadowRt)
#endif
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
// DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_BufferPtr(BlasGeom)), geometry_pointers)
// DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_BufferPtr(VoxelBrickAttribs)), attribute_pointers)
// DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VoxelBlasTransform), blas_transforms)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuVoxelObject), voxel_object_manifests)
DAXA_TH_TLAS_PTR(READ, tlas)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_3D, blue_noise_vec2)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, depth_image_id)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, particles_shadow_depth_tex)
// TODO: Figure out why this needs to be an ID...
DAXA_TH_IMAGE_ID(WRITE, REGULAR_2D, shadow_mask)
DAXA_DECL_TASK_HEAD_END
#if defined(DAXA_RAY_TRACING) || defined(__cplusplus)
struct TraceShadowRtPush {
    DAXA_TH_BLOB(TraceShadowRt, uses)
};
#endif

#if defined(__cplusplus)

#include <application/settings.hpp>

inline auto trace_shadows(GpuContext &gpu_context, GbufferDepth &gbuffer_depth, VoxelWorldBuffers &voxel_buffers, daxa::TaskImageView particles_shadow_depth_image) -> daxa::TaskImageView {
    auto shadow_mask = gpu_context.frame_task_graph.create_task_image({
        .format = daxa::Format::R8_UNORM,
        .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
        .name = "shadow_mask",
    });

    AppSettings::add<settings::Checkbox>({"Graphics", "Render Shadows", {.value = true}, {.task_graph_depends = true}});

    auto render_shadows = AppSettings::get<settings::Checkbox>("Graphics", "Render Shadows").value;

    if (render_shadows) {
#if USE_RAY_QUERY
        gpu_context.add(ComputeTask<TraceShadowRt::Info, TraceShadowRtPush, NoTaskInfo>{
#else
        gpu_context.add(RayTracingTask<TraceShadowRt::Info, TraceShadowRtPush, NoTaskInfo>{
#endif
            .source = "trace_shadow.rt.glsl",
            .views = TraceShadowRt::Views{
                .gpu_input = gpu_context.task_input_buffer.view(),
                // .geometry_pointers = voxel_buffers.blas_geom_pointers.task_resource.view(),
                // .attribute_pointers = voxel_buffers.blas_attr_pointers.task_resource.view(),
                // .blas_transforms = voxel_buffers.blas_transforms.task_resource.view(),
                .voxel_object_manifests = voxel_buffers.voxel_object_manifests.task_resource.view(),
                .tlas = voxel_buffers.task_tlas.view(),
                .blue_noise_vec2 = gpu_context.task_blue_noise_vec2_image.view(),
                .g_buffer_image_id = gbuffer_depth.gbuffer,
                .depth_image_id = gbuffer_depth.depth.current().view(),
                .particles_shadow_depth_tex = particles_shadow_depth_image,
                .shadow_mask = shadow_mask,
            },
#if USE_RAY_QUERY
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TraceShadowRtPush &push, NoTaskInfo const &) {
#else
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RayTracingPipeline &pipeline, daxa::RayTracingShaderBindingTable const &shader_binding_table, TraceShadowRtPush &push, NoTaskInfo const &) {
#endif
                auto const image_info = ti.device.image_info(ti.get(TraceShadowRt::AT.g_buffer_image_id).id).value();
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
#if USE_RAY_QUERY
                ti.recorder.dispatch({.x = round_up_div(image_info.size.x, 8), .y = round_up_div(image_info.size.y, 4), .z = 1});
#else
                ti.recorder.trace_rays({.width = image_info.size.x, .height = image_info.size.y, .depth = 1, .shader_binding_table = shader_binding_table});
#endif
            },
        });
    } else {
        clear_task_images(gpu_context.frame_task_graph, std::array<daxa::TaskImageView, 1>{shadow_mask}, std::array<daxa::ClearValue, 1>{std::array<float, 4>{1.0f, 1.0f, 1.0f, 1.0f}});
    }

    debug_utils::DebugDisplay::add_pass({.name = "trace shadow bitmap", .task_image_id = shadow_mask, .type = DEBUG_IMAGE_TYPE_DEFAULT_UINT});

    return daxa::TaskImageView{shadow_mask};
}

#endif

#endif // RENDERER_TRACE_SECONDARY_INL

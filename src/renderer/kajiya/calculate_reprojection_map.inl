#ifndef RENDERER_KAJIYA_CALCULATE_REPROJECTION_MAP_INL
#define RENDERER_KAJIYA_CALCULATE_REPROJECTION_MAP_INL

#include <core.inl>
#include <application/input.inl>
#include <renderer/core.inl>

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(CalculateReprojectionMapCompute)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, depth_image_id)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, prev_depth_image_id)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE_INDEX(WRITE, REGULAR_2D, dst_image_id)
DAXA_DECL_TASK_HEAD_END
struct CalculateReprojectionMapComputePush {
    DAXA_TH_BLOB(CalculateReprojectionMapCompute, uses)
};

#if defined(__cplusplus)

inline auto calculate_reprojection_map(GpuContext &gpu_context, GbufferDepth const &gbuffer_depth, daxa::TaskImageView velocity_image) -> daxa::TaskImageView {
    auto reprojection_map = gpu_context.frame_task_graph.create_task_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
        .name = "reprojection_image",
    });
    gpu_context.add(ComputeTask<CalculateReprojectionMapCompute::Info, CalculateReprojectionMapComputePush, NoTaskInfo>{
        .source = "kajiya/calculate_reprojection_map.comp.glsl",
        .views = CalculateReprojectionMapCompute::Views{
            .gpu_input = gpu_context.task_input_buffer.view(),
            .vs_normal_image_id = gbuffer_depth.geometric_normal,
            .depth_image_id = gbuffer_depth.depth.current().view(),
            .prev_depth_image_id = gbuffer_depth.depth.history().view(),
            .velocity_image_id = velocity_image,
            .dst_image_id = reprojection_map,
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, CalculateReprojectionMapComputePush &push, NoTaskInfo const &) {
            auto const image_info = ti.device.image_info(ti.get(CalculateReprojectionMapCompute::AT.dst_image_id).id).value();
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
        },
    });
    debug_utils::DebugDisplay::add_pass({.name = "reprojection_map", .task_image_id = reprojection_map, .type = DEBUG_IMAGE_TYPE_DEFAULT});
    return reprojection_map;
}

#endif

#endif // RENDERER_KAJIYA_CALCULATE_REPROJECTION_MAP_INL

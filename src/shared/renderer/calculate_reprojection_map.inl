#pragma once

#include <shared/core.inl>

#if CalculateReprojectionMapComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(CalculateReprojectionMapCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, prev_depth_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, dst_image_id)
DAXA_DECL_TASK_HEAD_END
struct CalculateReprojectionMapComputePush {
    CalculateReprojectionMapCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(CalculateReprojectionMapComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewId vs_normal_image_id = push.uses.vs_normal_image_id;
daxa_ImageViewId depth_image_id = push.uses.depth_image_id;
daxa_ImageViewId prev_depth_image_id = push.uses.prev_depth_image_id;
daxa_ImageViewId velocity_image_id = push.uses.velocity_image_id;
daxa_ImageViewId dst_image_id = push.uses.dst_image_id;
#endif
#endif

#if defined(__cplusplus)

inline auto calculate_reprojection_map(RecordContext &record_ctx, GbufferDepth const &gbuffer_depth, daxa::TaskImageView velocity_image) -> daxa::TaskImageView {
    auto reprojection_map = record_ctx.task_graph.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
        .name = "reprojection_image",
    });
    record_ctx.add(ComputeTask<CalculateReprojectionMapCompute, CalculateReprojectionMapComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"calculate_reprojection_map.comp.glsl"},
        .uses = {
            .gpu_input = record_ctx.task_input_buffer,
            .globals = record_ctx.task_globals_buffer,
            .vs_normal_image_id = gbuffer_depth.geometric_normal,
            .depth_image_id = gbuffer_depth.depth.task_resources.output_resource,
            .prev_depth_image_id = gbuffer_depth.depth.task_resources.history_resource,
            .velocity_image_id = velocity_image,
            .dst_image_id = reprojection_map,
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, CalculateReprojectionMapCompute::Uses &uses, CalculateReprojectionMapComputePush &push, NoTaskInfo const &) {
            auto const &image_info = ti.get_device().info_image(uses.dst_image_id.image()).value();
            ti.get_recorder().set_pipeline(pipeline);
            ti.get_recorder().push_constant(push);
            ti.get_recorder().dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
        },
    });
    AppUi::DebugDisplay::s_instance->passes.push_back({.name = "reprojection_map", .task_image_id = reprojection_map, .type = DEBUG_IMAGE_TYPE_DEFAULT});
    return reprojection_map;
}

#endif

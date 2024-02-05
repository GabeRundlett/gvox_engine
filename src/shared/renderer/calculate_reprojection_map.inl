#pragma once

#include <shared/core.inl>
#include <shared/input.inl>
#include <shared/globals.inl>
#include <shared/renderer/core.inl>

DAXA_DECL_TASK_HEAD_BEGIN(CalculateReprojectionMapCompute, 7)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, prev_depth_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, dst_image_id)
DAXA_DECL_TASK_HEAD_END
struct CalculateReprojectionMapComputePush {
    DAXA_TH_BLOB(CalculateReprojectionMapCompute, uses)
};

#if defined(__cplusplus)

inline auto calculate_reprojection_map(RecordContext &record_ctx, GbufferDepth const &gbuffer_depth, daxa::TaskImageView velocity_image) -> daxa::TaskImageView {
    auto reprojection_map = record_ctx.task_graph.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
        .name = "reprojection_image",
    });
    record_ctx.add(ComputeTask<CalculateReprojectionMapCompute, CalculateReprojectionMapComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"calculate_reprojection_map.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{CalculateReprojectionMapCompute::gpu_input, record_ctx.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{CalculateReprojectionMapCompute::globals, record_ctx.task_globals_buffer}},
            daxa::TaskViewVariant{std::pair{CalculateReprojectionMapCompute::vs_normal_image_id, gbuffer_depth.geometric_normal}},
            daxa::TaskViewVariant{std::pair{CalculateReprojectionMapCompute::depth_image_id, gbuffer_depth.depth.current()}},
            daxa::TaskViewVariant{std::pair{CalculateReprojectionMapCompute::prev_depth_image_id, gbuffer_depth.depth.history()}},
            daxa::TaskViewVariant{std::pair{CalculateReprojectionMapCompute::velocity_image_id, velocity_image}},
            daxa::TaskViewVariant{std::pair{CalculateReprojectionMapCompute::dst_image_id, reprojection_map}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, CalculateReprojectionMapComputePush &push, NoTaskInfo const &) {
            auto const image_info = ti.device.info_image(ti.get(CalculateReprojectionMapCompute::dst_image_id).ids[0]).value();
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
        },
    });
    AppUi::DebugDisplay::s_instance->passes.push_back({.name = "reprojection_map", .task_image_id = reprojection_map, .type = DEBUG_IMAGE_TYPE_DEFAULT});
    return reprojection_map;
}

#endif

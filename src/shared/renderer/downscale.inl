#pragma once

#include <shared/core.inl>

#if DownscaleComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(DownscaleCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, src_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, dst_image_id)
DAXA_DECL_TASK_HEAD_END
struct DownscaleComputePush {
    DownscaleCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(DownscaleComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewId src_image_id = push.uses.src_image_id;
daxa_ImageViewId dst_image_id = push.uses.dst_image_id;
#endif
#endif

#if defined(__cplusplus)

inline auto extract_downscaled_depth(RecordContext &record_ctx, daxa::TaskImageView depth) -> daxa::TaskImageView {
    auto size = record_ctx.render_resolution;

    auto output_tex = record_ctx.task_graph.create_transient_image({
        .format = daxa::Format::R32_SFLOAT,
        .size = {size.x / SHADING_SCL, size.y / SHADING_SCL, 1},
        .name = "downscaled_depth",
    });

    record_ctx.add(ComputeTask<DownscaleCompute, DownscaleComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"downscale.comp.glsl"},
        .extra_defines = {{"DOWNSCALE_DEPTH", "1"}},
        .uses = {
            .gpu_input = record_ctx.task_input_buffer,
            .globals = record_ctx.task_globals_buffer,
            .src_image_id = depth,
            .dst_image_id = output_tex,
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, DownscaleCompute::Uses &uses, DownscaleComputePush &push, NoTaskInfo const &) {
            auto const &image_info = ti.get_device().info_image(uses.dst_image_id.image()).value();
            ti.get_recorder().set_pipeline(pipeline);
            ti.get_recorder().push_constant(push);
            ti.get_recorder().dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
        },
    });

    return output_tex;
}

inline auto extract_downscaled_gbuffer_view_normal_rgba8(RecordContext &record_ctx, daxa::TaskImageView gbuffer) -> daxa::TaskImageView {
    auto size = record_ctx.render_resolution;

    auto output_tex = record_ctx.task_graph.create_transient_image({
        .format = daxa::Format::R8G8B8A8_SNORM,
        .size = {size.x / SHADING_SCL, size.y / SHADING_SCL, 1},
        .name = "downscaled_gbuffer_view_normal",
    });

    record_ctx.add(ComputeTask<DownscaleCompute, DownscaleComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"downscale.comp.glsl"},
        .extra_defines = {{"DOWNSCALE_NRM", "1"}},
        .uses = {
            .gpu_input = record_ctx.task_input_buffer,
            .globals = record_ctx.task_globals_buffer,
            .src_image_id = gbuffer,
            .dst_image_id = output_tex,
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, DownscaleCompute::Uses &uses, DownscaleComputePush &push, NoTaskInfo const &) {
            auto const &image_info = ti.get_device().info_image(uses.dst_image_id.image()).value();
            ti.get_recorder().set_pipeline(pipeline);
            ti.get_recorder().push_constant(push);
            ti.get_recorder().dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
        },
    });

    return output_tex;
}

#endif

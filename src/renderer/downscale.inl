#pragma once

#include <shared/core.inl>
#include <shared/input.inl>
#include <shared/globals.inl>

DAXA_DECL_TASK_HEAD_BEGIN(DownscaleCompute, 4)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, src_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, dst_image_id)
DAXA_DECL_TASK_HEAD_END
struct DownscaleComputePush {
    DAXA_TH_BLOB(DownscaleCompute, uses)
};

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
        .views = std::array{
            daxa::TaskViewVariant{std::pair{DownscaleCompute::gpu_input, record_ctx.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{DownscaleCompute::globals, record_ctx.task_globals_buffer}},
            daxa::TaskViewVariant{std::pair{DownscaleCompute::src_image_id, depth}},
            daxa::TaskViewVariant{std::pair{DownscaleCompute::dst_image_id, output_tex}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, DownscaleComputePush &push, NoTaskInfo const &) {
            auto const image_info = ti.device.info_image(ti.get(DownscaleCompute::dst_image_id).ids[0]).value();
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
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
        .views = std::array{
            daxa::TaskViewVariant{std::pair{DownscaleCompute::gpu_input, record_ctx.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{DownscaleCompute::globals, record_ctx.task_globals_buffer}},
            daxa::TaskViewVariant{std::pair{DownscaleCompute::src_image_id, gbuffer}},
            daxa::TaskViewVariant{std::pair{DownscaleCompute::dst_image_id, output_tex}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, DownscaleComputePush &push, NoTaskInfo const &) {
            auto const image_info = ti.device.info_image(ti.get(DownscaleCompute::dst_image_id).ids[0]).value();
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
        },
    });

    return output_tex;
}

inline auto extract_downscaled_ssao(RecordContext &record_ctx, daxa::TaskImageView ssao_tex) -> daxa::TaskImageView {
    auto size = record_ctx.render_resolution;

    auto output_tex = record_ctx.task_graph.create_transient_image({
        .format = daxa::Format::R8_SNORM,
        .size = {size.x / SHADING_SCL, size.y / SHADING_SCL, 1},
        .name = "downscaled_ssao",
    });

    record_ctx.add(ComputeTask<DownscaleCompute, DownscaleComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"downscale.comp.glsl"},
        .extra_defines = {{"DOWNSCALE_SSAO", "1"}},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{DownscaleCompute::gpu_input, record_ctx.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{DownscaleCompute::globals, record_ctx.task_globals_buffer}},
            daxa::TaskViewVariant{std::pair{DownscaleCompute::src_image_id, ssao_tex}},
            daxa::TaskViewVariant{std::pair{DownscaleCompute::dst_image_id, output_tex}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, DownscaleComputePush &push, NoTaskInfo const &) {
            auto const image_info = ti.device.info_image(ti.get(DownscaleCompute::dst_image_id).ids[0]).value();
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
        },
    });

    return output_tex;
}

#endif

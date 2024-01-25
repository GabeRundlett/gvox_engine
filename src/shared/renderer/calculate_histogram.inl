#pragma once

#include <shared/core.inl>

#define LUMINANCE_HISTOGRAM_BIN_COUNT 256
#define LUMINANCE_HISTOGRAM_MIN_LOG2 -6.0
#define LUMINANCE_HISTOGRAM_MAX_LOG2 10.0

#if CalculateHistogramComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(CalculateHistogramCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_tex)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), output_buffer)
DAXA_DECL_TASK_HEAD_END
struct CalculateHistogramComputePush {
    CalculateHistogramCompute uses;
    daxa_u32vec2 input_extent;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(CalculateHistogramComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId input_tex = push.uses.input_tex;
daxa_RWBufferPtr(daxa_u32) output_buffer = push.uses.output_buffer;
#endif
#endif

#if defined(__cplusplus)

inline auto calculate_luminance_histogram(RecordContext &record_ctx, daxa::TaskImageView blur_pyramid, daxa::TaskBufferView dst_histogram, daxa_u32vec2 image_size) {
    image_size = {(image_size.x + 1) / 2, (image_size.y + 1) / 2};
    auto mip_count = ceil_log2(std::max(image_size.x, image_size.y)) - 1;

    auto input_mip_level = std::max(mip_count, 7u) - 7;

    auto hist_size = static_cast<uint32_t>(sizeof(uint32_t) * LUMINANCE_HISTOGRAM_BIN_COUNT);
    auto tmp_histogram = record_ctx.task_graph.create_transient_buffer({
        .size = static_cast<uint32_t>(sizeof(uint32_t) * LUMINANCE_HISTOGRAM_BIN_COUNT),
        .name = "tmp_histogram",
    });

    record_ctx.task_graph.add_task({
        .uses = {
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{tmp_histogram},
        },
        .task = [=](daxa::TaskInterface ti) {
            auto &recorder = ti.get_recorder();
            recorder.clear_buffer({
                .buffer = ti.uses[tmp_histogram].buffer(),
                .offset = 0,
                .size = hist_size,
                .clear_value = 0,
            });
        },
        .name = "clear histogram",
    });

    struct CalculateHistogramTaskInfo {
        daxa_u32 input_mip_level;
    };
    record_ctx.add(ComputeTask<CalculateHistogramCompute, CalculateHistogramComputePush, CalculateHistogramTaskInfo>{
        .source = daxa::ShaderFile{"calculate_histogram.comp.glsl"},
        .uses = {
            .gpu_input = record_ctx.task_input_buffer,
            .input_tex = blur_pyramid.view({.base_mip_level = input_mip_level, .level_count = 1}),
            .output_buffer = tmp_histogram,
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, CalculateHistogramCompute::Uses &uses, CalculateHistogramComputePush &push, CalculateHistogramTaskInfo const &info) {
            auto const &image_info = ti.get_device().info_image(uses.input_tex.image()).value();
            push.input_extent = {(image_info.size.x + ((1 << info.input_mip_level) - 1)) >> info.input_mip_level, (image_info.size.y + ((1 << info.input_mip_level) - 1)) >> info.input_mip_level};
            ti.get_recorder().set_pipeline(pipeline);
            ti.get_recorder().push_constant(push);
            // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
            ti.get_recorder().dispatch({(push.input_extent.x + 7) / 8, (push.input_extent.y + 7) / 8});
        },
        .info = {
            .input_mip_level = input_mip_level,
        },
    });

    record_ctx.task_graph.add_task({
        .uses = {
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_READ>{tmp_histogram},
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{dst_histogram},
        },
        .task = [=](daxa::TaskInterface ti) {
            auto &recorder = ti.get_recorder();
            recorder.copy_buffer_to_buffer({
                .src_buffer = ti.uses[tmp_histogram].buffer(),
                .dst_buffer = ti.uses[dst_histogram].buffer(),
                .size = hist_size,
            });
        },
        .name = "copy histogram",
    });
}

#endif

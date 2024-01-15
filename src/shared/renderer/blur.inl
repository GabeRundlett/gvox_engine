#pragma once

#include <shared/core.inl>

#if BLUR_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(BlurCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, output_tex)
DAXA_DECL_TASK_HEAD_END
struct BlurComputePush {
    BlurCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(BlurComputePush, push)
daxa_ImageViewId input_tex = push.uses.input_tex;
daxa_ImageViewId output_tex = push.uses.output_tex;
#endif
#endif

#if REV_BLUR_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(RevBlurCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_tail_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, output_tex)
DAXA_DECL_TASK_HEAD_END
struct RevBlurComputePush {
    RevBlurCompute uses;
    daxa_u32vec4 output_extent;
    daxa_f32 self_weight;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(RevBlurComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId input_tail_tex = push.uses.input_tail_tex;
daxa_ImageViewId input_tex = push.uses.input_tex;
daxa_ImageViewId output_tex = push.uses.output_tex;
#endif
#endif

#if defined(__cplusplus)

struct BlurComputeTaskState {
    AsyncManagedComputePipeline pipeline;

    BlurComputeTaskState(AsyncPipelineManager &pipeline_manager) {
        pipeline = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"blur.comp.glsl"},
                .compile_options = {.defines = {{"BLUR_COMPUTE", "1"}}},
            },
            .push_constant_size = sizeof(BlurComputePush),
            .name = "blur",
        });
    }

    void record_commands(BlurComputePush const &push, daxa::CommandRecorder &recorder, daxa_u32vec2 render_size) {
        if (!pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(pipeline.get());
        recorder.push_constant(push);
        // assert((render_size.x % 64) == 0);
        recorder.dispatch({(render_size.x + 63) / 64, render_size.y});
    }
};
struct RevBlurComputeTaskState {
    AsyncManagedComputePipeline pipeline;

    RevBlurComputeTaskState(AsyncPipelineManager &pipeline_manager) {
        pipeline = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"blur.comp.glsl"},
                .compile_options = {.defines = {{"REV_BLUR_COMPUTE", "1"}}},
            },
            .push_constant_size = sizeof(RevBlurComputePush),
            .name = "rev_blur",
        });
    }

    void record_commands(RevBlurComputePush const &push, daxa::CommandRecorder &recorder, daxa_u32vec2 render_size) {
        if (!pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(pipeline.get());
        recorder.push_constant(push);
        // assert((render_size.x % 64) == 0);
        recorder.dispatch({(render_size.x + 7) / 8, (render_size.y + 7) / 8});
    }
};

struct BlurComputeTask {
    BlurCompute::Uses uses;
    std::string name = "BlurCompute";
    BlurComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto const &image_info = ti.get_device().info_image(uses.output_tex.image()).value();
        auto push = BlurComputePush{};
        ti.copy_task_head_to(&push.uses);
        state->record_commands(push, recorder, {image_info.size.x, image_info.size.y});
    }
};
struct RevBlurComputeTask {
    RevBlurCompute::Uses uses;
    std::string name = "RevBlurCompute";
    RevBlurComputeTaskState *state;
    daxa_u32 downsample_amount;
    daxa_f32 self_weight;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto const &image_info = ti.get_device().info_image(uses.output_tex.image()).value();
        auto push = RevBlurComputePush{};
        ti.copy_task_head_to(&push.uses);
        push.output_extent = {image_info.size.x / downsample_amount, image_info.size.y / downsample_amount};
        push.self_weight = self_weight;
        state->record_commands(push, recorder, {image_info.size.x, image_info.size.y});
    }
};

static constexpr auto ceil_log2(uint32_t x) -> uint32_t {
    constexpr auto const t = std::array<uint32_t, 5>{
        0xFFFF0000u,
        0x0000FF00u,
        0x000000F0u,
        0x0000000Cu,
        0x00000002u};

    uint32_t y = (((x & (x - 1)) == 0) ? 0 : 1);
    int j = 16;

    for (uint32_t const i : t) {
        int const k = (((x & i) == 0) ? 0 : j);
        y += static_cast<uint32_t>(k);
        x >>= k;
        j >>= 1;
    }

    return y;
}

inline auto blur_pyramid(RecordContext &record_ctx, BlurComputeTaskState &blur_task_state, daxa::TaskImageView input_image, daxa_u32vec2 image_size) -> daxa::TaskImageView {
    image_size = {(image_size.x + 1) / 2, (image_size.y + 1) / 2};
    auto mip_count = ceil_log2(std::max(image_size.x, image_size.y)) - 1;

    auto output = record_ctx.task_graph.create_transient_image({
        .format = daxa::Format::B10G11R11_UFLOAT_PACK32,
        .size = {image_size.x, image_size.y, 1},
        .mip_level_count = mip_count,
        .name = "blur_pyramid_output",
    });

    record_ctx.task_graph.add_task(BlurComputeTask{
        .uses = {
            .input_tex = input_image,
            .output_tex = output.view({.base_mip_level = 0, .level_count = 1}),
        },
        .state = &blur_task_state,
    });

    // AppUi::DebugDisplay::s_instance->passes.push_back({.name = "blur_pyramid mip 0", .task_image_id = output, .type = DEBUG_IMAGE_TYPE_DEFAULT});
    for (uint32_t mip_i = 0; mip_i < mip_count - 1; ++mip_i) {
        auto src = output.view({.base_mip_level = mip_i + 0, .level_count = 1});
        auto dst = output.view({.base_mip_level = mip_i + 1, .level_count = 1});
        record_ctx.task_graph.add_task(BlurComputeTask{
            .uses = {
                .input_tex = src,
                .output_tex = dst,
            },
            .state = &blur_task_state,
        });
        // AppUi::DebugDisplay::s_instance->passes.push_back({.name = "blur_pyramid mip " + std::to_string(mip_i + 1), .task_image_id = dst, .type = DEBUG_IMAGE_TYPE_DEFAULT});
    }

    return output;
}

inline auto rev_blur_pyramid(RecordContext &record_ctx, RevBlurComputeTaskState &rev_blur_task_state, daxa::TaskImageView input_image, daxa_u32vec2 image_size) -> daxa::TaskImageView {
    image_size = {(image_size.x + 1) / 2, (image_size.y + 1) / 2};
    auto mip_count = ceil_log2(std::max(image_size.x, image_size.y)) - 1;

    auto output = record_ctx.task_graph.create_transient_image({
        .format = daxa::Format::B10G11R11_UFLOAT_PACK32,
        .size = {image_size.x, image_size.y, 1},
        .mip_level_count = mip_count,
        .name = "rev_blur_pyramid_output",
    });

    for (uint32_t mip_i = 0; mip_i < mip_count - 1; ++mip_i) {
        auto target_mip_i = mip_count - mip_i - 2;
        auto downsample_amount = 1u << target_mip_i;
        auto self_weight = (target_mip_i + 1 == mip_count) ? 0.0f : 0.5f;

        auto tail = input_image.view({.base_mip_level = target_mip_i + 0, .level_count = 1});
        auto src = output.view({.base_mip_level = target_mip_i + 1, .level_count = 1});
        auto dst = output.view({.base_mip_level = target_mip_i + 0, .level_count = 1});

        record_ctx.task_graph.add_task(RevBlurComputeTask{
            .uses = {
                .input_tail_tex = tail,
                .input_tex = src,
                .output_tex = dst,
            },
            .state = &rev_blur_task_state,
            .downsample_amount = downsample_amount,
            .self_weight = self_weight,
        });
        // AppUi::DebugDisplay::s_instance->passes.push_back({.name = "rev blur_pyramid mip " + std::to_string(target_mip_i), .task_image_id = dst, .type = DEBUG_IMAGE_TYPE_DEFAULT});
    }

    return output;
}

#endif

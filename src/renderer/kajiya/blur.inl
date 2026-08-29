#ifndef RENDERER_KAJIYA_BLUR_INL
#define RENDERER_KAJIYA_BLUR_INL

#include <core.inl>
#include <application/input.inl>

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(BlurCompute)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, input_tex)
DAXA_TH_IMAGE_INDEX(WRITE, REGULAR_2D, output_tex)
DAXA_DECL_TASK_HEAD_END
struct BlurComputePush {
    DAXA_TH_BLOB(BlurCompute, uses)
};

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RevBlurCompute)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, input_tail_tex)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, input_tex)
DAXA_TH_IMAGE_INDEX(WRITE, REGULAR_2D, output_tex)
DAXA_DECL_TASK_HEAD_END
struct RevBlurComputePush {
    daxa_u32vec4 output_extent;
    daxa_f32 self_weight;
    DAXA_TH_BLOB(RevBlurCompute, uses)
};

#if defined(__cplusplus)

inline auto blur_pyramid(GpuContext &gpu_context, daxa::TaskImageView input_image, daxa_u32vec2 image_size) -> daxa::TaskImageView {
    image_size = {(image_size.x + 1) / 2, (image_size.y + 1) / 2};
    auto mip_count = ceil_log2(std::max(image_size.x, image_size.y)) - 1;

    auto output = gpu_context.frame_task_graph.create_task_image({
        .format = daxa::Format::B10G11R11_UFLOAT_PACK32,
        .size = {image_size.x, image_size.y, 1},
        .mip_level_count = mip_count,
        .name = "blur_pyramid_output",
    });

    struct BlurTaskInfo {
        daxa_u32 mip_i;
    };
    auto blur_dispatch = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, BlurComputePush &push, BlurTaskInfo const &info) {
        auto const image_info = ti.device.image_info(ti.get(BlurCompute::AT.output_tex).id).value();
        auto downscale_factor = 1u << info.mip_i;
        ti.recorder.set_pipeline(pipeline);
        set_push_constant(ti, push);
        ti.recorder.dispatch({((image_info.size.x + downscale_factor - 1) / downscale_factor + 63) / 64, (image_info.size.y + downscale_factor - 1) / downscale_factor});
    };
    gpu_context.add(ComputeTask<BlurCompute::Info, BlurComputePush, BlurTaskInfo>{
        .source = "kajiya/blur.comp.glsl",
        .views = BlurCompute::Views{
            .input_tex = input_image,
            .output_tex = output.mips(0, 1),
        },
        .callback_ = blur_dispatch,
        .info = {
            .mip_i = 0,
        },
    });

    #if 0
    // debug_utils::DebugDisplay::add_pass({.name = "blur_pyramid mip 0", .task_image_id = output, .type = DEBUG_IMAGE_TYPE_DEFAULT});
    for (uint32_t mip_i = 0; mip_i < mip_count - 1; ++mip_i) {
        auto src = output.mips(mip_i + 0, 1);
        auto dst = output.mips(mip_i + 1, 1);
        gpu_context.add(ComputeTask<BlurCompute::Info, BlurComputePush, BlurTaskInfo>{
            .source = "kajiya/blur.comp.glsl",
            .views = BlurCompute::Views{
                .input_tex = src,
                .output_tex = dst,
            },
            .callback_ = blur_dispatch,
            .info = {
                .mip_i = mip_i + 1,
            },
        });
        // debug_utils::DebugDisplay::add_pass({.name = "blur_pyramid mip " + std::to_string(mip_i + 1), .task_image_id = dst, .type = DEBUG_IMAGE_TYPE_DEFAULT});
    }
    #endif

    return output;
}

inline auto rev_blur_pyramid(GpuContext &gpu_context, daxa::TaskImageView input_image, daxa_u32vec2 image_size) -> daxa::TaskImageView {
    image_size = {(image_size.x + 1) / 2, (image_size.y + 1) / 2};
    auto mip_count = ceil_log2(std::max(image_size.x, image_size.y)) - 1;

    auto output = gpu_context.frame_task_graph.create_task_image({
        .format = daxa::Format::B10G11R11_UFLOAT_PACK32,
        .size = {image_size.x, image_size.y, 1},
        .mip_level_count = mip_count,
        .name = "rev_blur_pyramid_output",
    });

    for (uint32_t mip_i = 0; mip_i < mip_count - 1; ++mip_i) {
        auto target_mip_i = mip_count - mip_i - 2;
        auto downsample_amount = 1u << target_mip_i;
        auto self_weight = (target_mip_i + 1 == mip_count) ? 0.0f : 0.5f;

        auto tail = input_image.mips(target_mip_i + 0, 1);
        auto src = output.mips(target_mip_i + 1, 1);
        auto dst = output.mips(target_mip_i + 0, 1);

        struct RevBlurTaskInfo {
            daxa_u32 downsample_amount;
            daxa_f32 self_weight;
        };
        gpu_context.add(ComputeTask<RevBlurCompute::Info, RevBlurComputePush, RevBlurTaskInfo>{
            .source = "kajiya/blur.comp.glsl",
            .views = RevBlurCompute::Views{
                .gpu_input = gpu_context.task_input_buffer.view(),
                .input_tail_tex = tail,
                .input_tex = src,
                .output_tex = dst,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RevBlurComputePush &push, RevBlurTaskInfo const &info) {
                auto const image_info = ti.device.image_info(ti.get(RevBlurCompute::AT.output_tex).id).value();
                push.output_extent = {image_info.size.x / info.downsample_amount, image_info.size.y / info.downsample_amount, 1};
                push.self_weight = info.self_weight;
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
                ti.recorder.dispatch({(push.output_extent.x + 7) / 8, (push.output_extent.y + 7) / 8});
            },
            .info = {
                .downsample_amount = downsample_amount,
                .self_weight = self_weight,
            },
        });
        // debug_utils::DebugDisplay::add_pass({.name = "rev blur_pyramid mip " + std::to_string(target_mip_i), .task_image_id = dst, .type = DEBUG_IMAGE_TYPE_DEFAULT});
    }

    return output;
}

#endif

#endif // RENDERER_KAJIYA_BLUR_INL

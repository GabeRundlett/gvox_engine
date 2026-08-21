#pragma once

#include <renderer/core.inl>

#define GEN_HIZ_X 16
#define GEN_HIZ_Y 16
#define GEN_HIZ_LEVELS_PER_DISPATCH 12
#define GEN_HIZ_WINDOW_X 64
#define GEN_HIZ_WINDOW_Y 64
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(GenHiz)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(SAMPLE, REGULAR_2D, src)
DAXA_TH_IMAGE_ID_MIP_ARRAY(READ_WRITE, REGULAR_2D, mips, GEN_HIZ_LEVELS_PER_DISPATCH)
DAXA_DECL_TASK_HEAD_END

struct GenHizPush {
    DAXA_TH_BLOB(GenHiz, uses)
    daxa_RWBufferPtr(daxa_u32) counter;
    daxa_u32 mip_count;
    daxa_u32 total_workgroup_count;
};

#if defined(__cplusplus)

#include "gpu_context.hpp"

auto task_gen_hiz_single_pass(GpuContext &gpu_context, daxa::TaskGraph &task_graph, daxa::TaskImageView depth) -> daxa::TaskImageView {
    auto const x_ = gpu_context.next_lower_po2_render_size.x;
    auto const y_ = gpu_context.next_lower_po2_render_size.y;
    auto mip_count = static_cast<uint32_t>(std::ceil(std::log2(std::max(x_, y_))));
    mip_count = std::min(mip_count, uint32_t(GEN_HIZ_LEVELS_PER_DISPATCH));
    auto task_hiz = task_graph.create_task_image({
        .format = daxa::Format::R32_SFLOAT,
        .size = {x_, y_, 1},
        .mip_level_count = mip_count,
        .array_layer_count = 1,
        .sample_count = 1,
        .name = "hiz",
    });

    gpu_context.add(ComputeTask<GenHiz::Info, GenHizPush, NoTaskInfo>{
        .source = "gen_hiz.glsl",
        .views = GenHiz::Views{
            .gpu_input = gpu_context.task_input_buffer.view(),
            .src = depth,
            .mips = task_hiz,
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, GenHizPush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            auto const &image_attach_info = ti.get(GenHiz::AT.src);
            auto image_info = ti.device.info(image_attach_info.id).value();
            auto const dispatch_x = round_up_div(find_next_lower_po2(image_info.size.x) * 2, GEN_HIZ_WINDOW_X);
            auto const dispatch_y = round_up_div(find_next_lower_po2(image_info.size.y) * 2, GEN_HIZ_WINDOW_Y);
            push = {
                .counter = ti.allocator->allocate_fill(0u).value().device_address,
                .mip_count = ti.get(GenHiz::AT.mips).view.slice.level_count,
                .total_workgroup_count = dispatch_x * dispatch_y,
            };
            set_push_constant(ti, push);
            ti.recorder.dispatch({dispatch_x, dispatch_y, 1});
        },
        .task_graph_ptr = &task_graph,
    });

    return task_hiz;
}

#endif
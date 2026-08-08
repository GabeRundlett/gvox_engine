#pragma once

#include <application/input.inl>

DAXA_DECL_RASTER_TASK_HEAD_BEGIN(R32D32Blit)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, input_tex)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, output_tex)
DAXA_DECL_TASK_HEAD_END
struct R32D32BlitPush {
    DAXA_TH_BLOB(R32D32Blit, uses)
};

#if defined(__cplusplus)

#include <utilities/gpu_context.hpp>
#include <renderer/kajiya/gbuffer.hpp>

namespace {
    template <size_t N>
    inline void clear_task_images(daxa::TaskGraph &task_graph, std::array<daxa::TaskImageView, N> const &task_image_views, std::array<daxa::ClearValue, N> clear_values = {}) {
        auto use_count = task_image_views.size();
        auto task = daxa::InlineTask::Transfer("clear images");
        for (auto const &task_image : task_image_views) {
            task.writes(daxa::ImageViewType::REGULAR_2D, task_image);
        }
        task.executes([use_count, clear_values](daxa::TaskInterface const &ti) {
            for (uint8_t i = 0; i < use_count; ++i) {
                ti.recorder.clear_image({
                    .image = ti.get(daxa::TaskImageAttachmentIndex{i}).id,
                    .clear_value = clear_values[i],
                });
            }
        });
        task_graph.add_task(task);
    }
    template <size_t N>
    inline void clear_task_images(daxa::Device &device, std::array<daxa::ExternalTaskImage, N> const &task_images) {
        daxa::TaskGraph temp_task_graph = daxa::TaskGraph({
            .device = device,
            .name = "temp_task_graph",
        });
        auto task_image_views = std::array<daxa::TaskImageView, N>{};
        for (size_t i = 0; i < N; ++i) {
            task_image_views[i] = task_images[i];
            temp_task_graph.register_image(task_images[i]);
        }
        clear_task_images(temp_task_graph, task_image_views);
        temp_task_graph.submit({});
        temp_task_graph.complete({});
        temp_task_graph.execute({});
    }

    auto extent_inv_extent_2d(daxa::ImageInfo const &image_info) -> daxa_f32vec4 {
        auto result = daxa_f32vec4{};
        result.x = static_cast<float>(image_info.size.x);
        result.y = static_cast<float>(image_info.size.y);
        result.z = 1.0f / result.x;
        result.w = 1.0f / result.y;
        return result;
    }

    void r32_d32_blit(GpuContext& gpu_context, daxa::TaskImageView src, daxa::TaskImageView dst) {
        gpu_context.add(RasterTask<R32D32Blit::Info, R32D32BlitPush, NoTaskInfo>{
            .vert_source = daxa::ShaderFile{"FULL_SCREEN_TRIANGLE_VERTEX_SHADER"},
            .frag_source = daxa::ShaderFile{"R32_D32_BLIT"},
            .depth_test = daxa::DepthTestInfo{
                .depth_attachment_format = daxa::Format::D32_SFLOAT,
                .enable_depth_write = true,
                .depth_test_compare_op = daxa::CompareOp::ALWAYS,
            },
            .views = R32D32Blit::Views{
                .input_tex = src,
                .output_tex = dst,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, R32D32BlitPush &push, NoTaskInfo const &) {
                auto render_image = ti.get(R32D32Blit::AT.output_tex).id;
                auto const image_info = ti.device.image_info(render_image).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .depth_attachment = {{.image_view = ti.get(R32D32Blit::AT.output_tex).view_ids[0], .load_op = daxa::AttachmentLoadOp::CLEAR, .clear_value = std::array{0.0f, 0.0f, 0.0f, 0.0f}}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.draw({.vertex_count = 3});
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });
    }
} // namespace

#endif

#pragma once

#include <application/input.inl>

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
} // namespace

#endif

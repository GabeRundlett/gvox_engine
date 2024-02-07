#pragma once

#include <renderer/downscale.inl>

#if defined(__cplusplus)

#include <cpu/core.hpp>

struct GbufferDepth {
    daxa::TaskImageView gbuffer;
    daxa::TaskImageView geometric_normal;
    PingPongImage depth;

    std::optional<daxa::TaskImageView> downscaled_depth = std::nullopt;
    std::optional<daxa::TaskImageView> downscaled_view_normal = std::nullopt;

    void next_frame() {
        depth.swap();
        downscaled_depth = std::nullopt;
        downscaled_view_normal = std::nullopt;
    }

    auto get_downscaled_depth(RecordContext &ctx) -> daxa::TaskImageView {
        if (!downscaled_depth) {
            downscaled_depth = extract_downscaled_depth(ctx, depth.current());
        }
        return *downscaled_depth;
    }
    auto get_downscaled_view_normal(RecordContext &ctx) -> daxa::TaskImageView {
        if (!downscaled_view_normal) {
            downscaled_view_normal = extract_downscaled_gbuffer_view_normal_rgba8(ctx, gbuffer);
        }
        return *downscaled_view_normal;
    }
};

namespace {
    template <size_t N>
    inline void clear_task_images(daxa::TaskGraph &task_graph, std::array<daxa::TaskImageView, N> const &task_image_views) {
        auto uses = std::vector<daxa::TaskAttachmentInfo>{};
        auto use_count = task_image_views.size();
        uses.reserve(use_count);
        for (auto const &task_image : task_image_views) {
            uses.push_back(daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, daxa::ImageViewType::REGULAR_2D, task_image));
        }
        task_graph.add_task({
            .attachments = std::move(uses),
            .task = [use_count](daxa::TaskInterface const &ti) {
                for (uint8_t i = 0; i < use_count; ++i) {
                    ti.recorder.clear_image({
                        .dst_image_layout = ti.get(daxa::TaskImageAttachmentIndex{i}).layout,
                        .dst_image = ti.get(daxa::TaskImageAttachmentIndex{i}).ids[0],
                    });
                }
            },
            .name = "clear images",
        });
    }
    template <size_t N>
    inline void clear_task_images(daxa::Device &device, std::array<daxa::TaskImage, N> const &task_images) {
        daxa::TaskGraph temp_task_graph = daxa::TaskGraph({
            .device = device,
            .name = "temp_task_graph",
        });
        auto task_image_views = std::array<daxa::TaskImageView, N>{};
        for (size_t i = 0; i < N; ++i) {
            task_image_views[i] = task_images[i];
            temp_task_graph.use_persistent_image(task_images[i]);
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

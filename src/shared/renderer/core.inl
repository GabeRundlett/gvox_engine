#pragma once

#include <shared/renderer/downscale.inl>

#if defined(__cplusplus)

#include <cpu/core.hpp>

struct GbufferDepth {
    daxa::TaskImageView gbuffer;
    daxa::TaskImageView geometric_normal;
    PingPongImage depth;

    std::optional<daxa::TaskImageView> downscaled_depth = std::nullopt;
    std::optional<daxa::TaskImageView> downscaled_view_normal = std::nullopt;

    void next_frame() {
        depth.task_resources.output_resource.swap_images(depth.task_resources.history_resource);
        downscaled_depth = std::nullopt;
        downscaled_view_normal = std::nullopt;
    }

    auto get_downscaled_depth(RecordContext &ctx) -> daxa::TaskImageView {
        if (!downscaled_depth) {
            downscaled_depth = extract_downscaled_depth(ctx, depth.task_resources.output_resource);
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
    inline void clear_task_images(daxa::Device &device, std::array<daxa::TaskImage, N> task_images) {
        daxa::TaskGraph temp_task_graph = daxa::TaskGraph({
            .device = device,
            .name = "temp_task_graph",
        });
        auto uses = std::vector<daxa::GenericTaskResourceUse>{};
        auto views = std::vector<daxa::TaskImageView>{};
        uses.reserve(task_images.size());
        views.reserve(task_images.size());
        for (auto const &task_image : task_images) {
            temp_task_graph.use_persistent_image(task_image);
            uses.push_back(daxa::TaskImageUse<daxa::TaskImageAccess::TRANSFER_WRITE>{task_image});
            views.push_back(task_image);
        }
        temp_task_graph.add_task({
            .uses = std::move(uses),
            .task = [&views](daxa::TaskInterface ti) {
                auto &recorder = ti.get_recorder();
                for (auto const &view : views) {
                    recorder.clear_image({
                        .dst_image_layout = ti.uses[view].layout(),
                        .dst_image = ti.uses[view].image(),
                    });
                }
            },
            .name = "clear images",
        });
        temp_task_graph.submit({});
        temp_task_graph.complete({});
        temp_task_graph.execute({});
    }
} // namespace

#endif

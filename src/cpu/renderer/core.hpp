#pragma once

#include <cpu/core.hpp>
#include <shared/tasks/renderer/downscale.inl>

struct PingPongImage {
    struct Resources {
        Resources() = default;
        Resources(Resources const &) = delete;
        Resources(Resources &&) = delete;
        Resources &operator=(Resources const &) = delete;
        Resources &operator=(Resources &&other) {
            std::swap(this->device, other.device);
            std::swap(this->image_a, other.image_a);
            std::swap(this->image_b, other.image_b);
            return *this;
        }

        daxa::Device device{};
        daxa::ImageId image_a{};
        daxa::ImageId image_b{};

        ~Resources() {
            if (!image_a.is_empty()) {
                device.destroy_image(image_a);
                device.destroy_image(image_b);
            }
        }
    };
    struct TaskResources {
        daxa::TaskImage output_image;
        daxa::TaskImage history_image;
    };
    Resources resources;
    TaskResources task_resources;

    auto get(daxa::Device a_device, daxa::ImageInfo const &a_info) -> std::pair<daxa::TaskImage &, daxa::TaskImage &> {
        if (!resources.device) {
            resources.device = a_device;
        }
        assert(resources.device == a_device);
        if (resources.image_a.is_empty()) {
            auto info_a = a_info;
            auto info_b = a_info;
            info_a.name += "_a";
            info_b.name += "_b";
            resources.image_a = a_device.create_image(info_a);
            resources.image_b = a_device.create_image(info_b);
            task_resources.output_image = daxa::TaskImage(daxa::TaskImageInfo{
                .initial_images = {.images = std::array{resources.image_a}},
                .name = a_info.name,
            });
            task_resources.history_image = daxa::TaskImage(daxa::TaskImageInfo{
                .initial_images = {.images = std::array{resources.image_b}},
                .name = a_info.name + "_history",
            });
        }
        return {task_resources.output_image, task_resources.history_image};
    }
};

struct GbufferDepth {
    daxa::TaskImageView gbuffer;
    daxa::TaskImageView geometric_normal;
    PingPongImage depth;

    // DownscaleComputeTaskState downscale_normal_task_state{};
    DownscaleComputeTaskState2 downscale_depth_task_state{};
    // std::optional<daxa::TaskImageView> downscaled_view_normal = std::nullopt;
    std::optional<daxa::TaskImageView> downscaled_depth = std::nullopt;

    void next_frame() {
        depth.task_resources.output_image.swap_images(depth.task_resources.history_image);
        // downscaled_view_normal = std::nullopt;
        downscaled_depth = std::nullopt;
    }

    // auto get_downscaled_view_normal(RecordContext &ctx) -> daxa::TaskImageView {
    //     if (!downscaled_view_normal) {
    //         if (!downscale_normal_task_state.pipeline) {
    //             downscale_normal_task_state = DownscaleComputeTaskState(*ctx.pipeline_manager, {{"DOWNSCALE_DEPTH", "1"}});
    //         }
    //         downscaled_view_normal = extract_downscaled_gbuffer_view_normal_rgba8(ctx, downscale_normal_task_state, gbuffer, ctx.render_resolution);
    //     }
    //     return *downscaled_view_normal;
    // }
    auto get_downscaled_depth(RecordContext &ctx) -> daxa::TaskImageView {
        if (!downscaled_depth) {
            if (!downscale_depth_task_state.pipeline) {
                downscale_depth_task_state = DownscaleComputeTaskState2(ctx.pipeline_manager, {{"DOWNSCALE_NRM", "1"}});
            }
            downscaled_depth = extract_downscaled_depth(ctx, downscale_depth_task_state, depth.task_resources.output_image, ctx.render_resolution);
        }
        return *downscaled_depth;
    }
};

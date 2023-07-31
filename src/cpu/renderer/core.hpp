#pragma once

#include <cpu/core.hpp>
#include <shared/tasks/renderer/downscale.inl>

struct GbufferDepth {
    daxa::TaskImageView gbuffer;
    daxa::TaskImageView geometric_normal;
    PingPongImage depth;

    DownscaleComputeTaskState downscale_depth_task_state;
    DownscaleComputeTaskState downscale_normal_task_state;
    std::optional<daxa::TaskImageView> downscaled_depth = std::nullopt;
    std::optional<daxa::TaskImageView> downscaled_view_normal = std::nullopt;

    GbufferDepth(daxa::PipelineManager &pipeline_manager)
        : downscale_depth_task_state{pipeline_manager, {{"DOWNSCALE_DEPTH", "1"}}},
          downscale_normal_task_state{pipeline_manager, {{"DOWNSCALE_NRM", "1"}}} {
    }

    void next_frame() {
        depth.task_resources.output_image.swap_images(depth.task_resources.history_image);
        downscaled_depth = std::nullopt;
        downscaled_view_normal = std::nullopt;
    }

    auto get_downscaled_depth(RecordContext &ctx) -> daxa::TaskImageView {
        if (!downscaled_depth) {
            downscaled_depth = extract_downscaled_depth(ctx, downscale_depth_task_state, depth.task_resources.output_image);
        }
        return *downscaled_depth;
    }
    auto get_downscaled_view_normal(RecordContext &ctx) -> daxa::TaskImageView {
        if (!downscaled_view_normal) {
            downscaled_view_normal = extract_downscaled_gbuffer_view_normal_rgba8(ctx, downscale_normal_task_state, gbuffer);
        }
        return *downscaled_view_normal;
    }
};

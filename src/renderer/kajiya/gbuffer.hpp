#pragma once

#include <renderer/kajiya/downscale.inl>

#include <utilities/gpu_context.hpp>
#include <utilities/ping_pong_resource.hpp>

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

    auto get_downscaled_depth(GpuContext &gpu_context) -> daxa::TaskImageView {
        if (!downscaled_depth) {
            downscaled_depth = extract_downscaled_depth(gpu_context, depth.current());
        }
        return *downscaled_depth;
    }
    auto get_downscaled_view_normal(GpuContext &gpu_context) -> daxa::TaskImageView {
        if (!downscaled_view_normal) {
            downscaled_view_normal = extract_downscaled_gbuffer_view_normal_rgba8(gpu_context, gbuffer);
        }
        return *downscaled_view_normal;
    }
};

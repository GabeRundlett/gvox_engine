#pragma once

#include <renderer/kajiya/downscale.inl>

#include <utilities/record_context.hpp>
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

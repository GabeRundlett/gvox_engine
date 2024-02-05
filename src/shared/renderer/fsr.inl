#pragma once

#include <shared/core.inl>
#include <shared/renderer/core.inl>

#if defined(__cplusplus)
#include <ffx-fsr2-api/ffx_fsr2.h>

struct Fsr2Info {
    daxa_u32vec2 render_resolution = {};
    daxa_u32vec2 display_resolution = {};
};

struct Fsr2State {
    bool should_reset = {};
    daxa_f32 delta_time = {};
    daxa_f32vec2 jitter = {};

    bool should_sharpen = {};
    daxa_f32 sharpening = 0.0f;

    struct CameraInfo {
        daxa_f32 near_plane = {};
        daxa_f32 far_plane = {};
        daxa_f32 vertical_fov = {};
    };
    CameraInfo camera_info = {};
};

struct Fsr2Renderer {
    daxa::Device device;
    Fsr2State state;
    Fsr2Info info;
    size_t jitter_frame_i;

    FfxFsr2Context fsr_context = {};
    FfxFsr2ContextDescription context_description = {};
    std::vector<std::byte> scratch_buffer = {};

    Fsr2Renderer() = delete;
    Fsr2Renderer(Fsr2Renderer &&other) = delete;
    Fsr2Renderer &operator=(Fsr2Renderer &&other) = delete;
    Fsr2Renderer(Fsr2Renderer const &) = delete;
    Fsr2Renderer &operator=(Fsr2Renderer const &) = delete;

    Fsr2Renderer(daxa::Device a_device, Fsr2Info a_info);
    ~Fsr2Renderer();
    void next_frame();
    auto upscale(RecordContext &record_ctx, GbufferDepth const &gbuffer_depth, daxa::TaskImageView color_image, daxa::TaskImageView velocity_image) -> daxa::TaskImageView;
};

#endif

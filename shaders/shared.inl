#pragma once

#include <daxa/daxa.inl>

DAXA_DECL_BUFFER_STRUCT(GpuInput, {
    u32vec2 frame_dim;
    f32vec2 view_origin;
    f32vec2 mouse_pos;
    f32 zoom;
    f32 time;
    i32 max_steps;
});

struct ComputePush {
    ImageViewId image_id;
    BufferRef(GpuInput) gpu_input;
};

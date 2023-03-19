#pragma once

#include <shared/core.inl>

struct GpuOutput {
    f32vec3 player_pos;
    u32 heap_size;
};
DAXA_ENABLE_BUFFER_PTR(GpuOutput)

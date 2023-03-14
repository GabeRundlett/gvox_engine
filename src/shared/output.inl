#pragma once

#include <shared/core.inl>

struct GpuOutput {
    u32 heap_size;
    f32vec3 player_pos;
};
DAXA_ENABLE_BUFFER_PTR(GpuOutput)

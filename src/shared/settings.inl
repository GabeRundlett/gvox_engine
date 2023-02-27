#pragma once

#include <shared/core.inl>

struct GpuSettings {
    f32 fov;
    f32 sensitivity;

    u32 log2_chunks_per_axis;
};
DAXA_ENABLE_BUFFER_PTR(GpuSettings)

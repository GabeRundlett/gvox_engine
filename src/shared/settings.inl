#pragma once

#include <shared/core.inl>

struct GpuSettings {
    f32 fov;
    f32 sensitivity;

    u32 log2_chunks_per_axis;
    u32 gpu_heap_size;
};
DAXA_ENABLE_BUFFER_PTR(GpuSettings)

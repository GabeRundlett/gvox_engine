#pragma once

#include <shared/core.inl>

struct GpuGvoxModel {
    u32 magic;
    i32 offset_x;
    i32 offset_y;
    i32 offset_z;
    u32 extent_x;
    u32 extent_y;
    u32 extent_z;
    u32 blob_size;
    u32 channel_flags;
    u32 channel_n;
    u32 data[1];
};
DAXA_DECL_BUFFER_PTR(GpuGvoxModel)

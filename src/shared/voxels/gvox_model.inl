#pragma once

#include <shared/core.inl>

struct GpuGvoxModel {
    daxa_u32 magic;
    daxa_i32 offset_x;
    daxa_i32 offset_y;
    daxa_i32 offset_z;
    daxa_u32 extent_x;
    daxa_u32 extent_y;
    daxa_u32 extent_z;
    daxa_u32 blob_size;
    daxa_u32 channel_flags;
    daxa_u32 channel_n;
    daxa_u32 data[1];
};
DAXA_DECL_BUFFER_PTR(GpuGvoxModel)

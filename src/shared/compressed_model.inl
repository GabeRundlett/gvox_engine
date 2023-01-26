#pragma once

#include <daxa/daxa.inl>

struct GpuGVox_RegionHeader {
    u32 variant_n;
    u32 blob_offset; // if variant_n == 1, this is just the voxel
};

struct GpuGVox_NodeHeader {
    u32 node_full_size;
    u32 region_count_x;
    u32 region_count_y;
    u32 region_count_z;
};

struct GpuGVox_CompressedModel {
    u32 node_n;
    GpuGVox_NodeHeader node_header;
    u32 data[1];
};

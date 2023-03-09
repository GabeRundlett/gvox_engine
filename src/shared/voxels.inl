#pragma once

#include <shared/core.inl>

#define CHUNK_SIZE 64

#define PALETTE_REGION_SIZE 8

#if PALETTE_REGION_SIZE == 8
#define PALETTE_MAX_COMPRESSED_VARIANT_N 367
#elif PALETTE_REGION_SIZE == 16
#define PALETTE_MAX_COMPRESSED_VARIANT_N 2559
#else
#error Unsupported Palette Region Size
#endif

#define PALETTES_PER_CHUNK_AXIS (CHUNK_SIZE / PALETTE_REGION_SIZE)
#define PALETTES_PER_CHUNK (PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS)

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
    u32 data[1 << 28];
};
DAXA_ENABLE_BUFFER_PTR(GpuGvoxModel)

struct VoxelChunkUniformity {
    u32 lod_x2[1024];
    u32 lod_x4[256];
    u32 lod_x8[64];
    u32 lod_x16[16];
    u32 lod_x32[4];
};

struct PaletteHeader {
    u32 variant_n;
    u32 blob_offset;
};

struct VoxelChunk {
    u32 edit_stage;
    VoxelChunkUniformity uniformity;
    PaletteHeader palette_headers[PALETTES_PER_CHUNK];
};
DAXA_ENABLE_BUFFER_PTR(VoxelChunk)

struct TempVoxel {
    u32 col_and_id;
    // u32 nrm;
};

struct TempVoxelChunk {
    TempVoxel voxels[CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE];
};
DAXA_ENABLE_BUFFER_PTR(TempVoxelChunk)

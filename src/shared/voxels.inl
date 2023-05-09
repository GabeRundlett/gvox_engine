#pragma once

#include <shared/voxel_malloc.inl>

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

// 1364 u32's
// 10.65625 bytes per 8x8x8
struct VoxelChunkUniformity {
    u32 lod_x2[1024];
    u32 lod_x4[256];
    u32 lod_x8[64];
    u32 lod_x16[16];
    u32 lod_x32[4];
};

// 8 bytes per 8x8x8
struct PaletteHeader {
    u32 variant_n;
    VoxelMalloc_Pointer blob_ptr;
};

struct VoxelChunk {
    u32 edit_stage;
    VoxelChunkUniformity uniformity;
    VoxelMalloc_ChunkLocalPageSubAllocatorState sub_allocator_state;
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

struct VoxelParentChunk {
    u32 children[512];
    u32 is_pointer[16];
};
DAXA_ENABLE_BUFFER_PTR(VoxelParentChunk)

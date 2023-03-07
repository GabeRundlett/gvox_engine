#pragma once

#include <shared/core.inl>

#define CHUNK_SIZE 64

struct VoxelChunkUniformity {
    u32 lod_x2[1024];
    u32 lod_x4[256];
    u32 lod_x8[64];
    u32 lod_x16[16];
    u32 lod_x32[4];
};

struct VoxelChunk {
    u32 edit_stage;
    VoxelChunkUniformity uniformity;
};
DAXA_ENABLE_BUFFER_PTR(VoxelChunk)

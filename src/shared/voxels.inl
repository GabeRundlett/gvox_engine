#pragma once

#include <shared/shapes.inl>

#define CHUNK_SIZE 64
#define VOXEL_SCL 8

#define CHUNK_NX 12
#define CHUNK_NY 12
#define CHUNK_NZ 4
#define CHUNK_N (CHUNK_NX * CHUNK_NY * CHUNK_NZ)
#define BLOCK_NX (CHUNK_NX * CHUNK_SIZE)
#define BLOCK_NY (CHUNK_NY * CHUNK_SIZE)
#define BLOCK_NZ (CHUNK_NZ * CHUNK_SIZE)
#define BLOCK_N (CHUNK_N * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE)

struct Voxel {
    f32vec3 col;
    f32vec3 nrm;
    u32 block_id;
};
struct PackedVoxel {
    u32 data;
};

struct WorldgenState {
    f32 t_noise;
    f32 r, r_xy;
    u32 block_id;
};
struct SurroundingInfo {
    u32 above_ids[15];
    u32 below_ids[15];
    u32 depth_above;
    u32 depth_below;
    u32 under_water;
    u32 above_water;
    f32vec3 nrm;
};

struct ChunkGenState {
    u32 edit_stage;
};

struct VoxelChunk {
    Box box;
    PackedVoxel packed_voxels[CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE];
};

struct VoxelUniformityChunk {
    u32 lod_x2[1024];
    u32 lod_x4[256];
    u32 lod_x8[64];
    u32 lod_x16[16];
    u32 lod_x32[4];
};

struct VoxelWorld {
    Box box;
    i32vec3 chunkgen_i;
    i32vec3 chunkgen_i2;
    f32vec3 center_pt;
    f32 last_update_time;
    VoxelChunk voxel_chunks[CHUNK_N];
    VoxelUniformityChunk uniformity_chunks[CHUNK_N];
    ChunkGenState chunks_genstate[CHUNK_N];
};

#pragma once

#include <shared/shapes.inl>

#define CHUNK_SIZE 64
#define VOXEL_SCL 8

#define USING_BRICKMAP 0

#define BRICKMAP_SIZE 8
#define BRICKMAP_SUBBRICK_N (BRICKMAP_SIZE * BRICKMAP_SIZE * BRICKMAP_SIZE)
#define BRICKMAP_MAX_N 262144

#define CHUNK_NX 8
#define CHUNK_NY 8
#define CHUNK_NZ 4
#define CHUNK_N (CHUNK_NX * CHUNK_NY * CHUNK_NZ)
#define BLOCK_NX (CHUNK_NX * CHUNK_SIZE)
#define BLOCK_NY (CHUNK_NY * CHUNK_SIZE)
#define BLOCK_NZ (CHUNK_NZ * CHUNK_SIZE)
#define BLOCK_N (CHUNK_N * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE)

#define MAX_CHUNK_UPDATES 64

const u32 CHUNK_PTR_U32_N = (CHUNK_N / 32 == 0) ? 1 : CHUNK_N / 32;

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

struct VoxelBrick {
    u32 has_subbricks[BRICKMAP_SUBBRICK_N / 32];
    PackedVoxel subbricks[BRICKMAP_SUBBRICK_N];
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
    f32vec3 center_pt;
    f32 last_update_time;
    u32 chunk_update_indices[MAX_CHUNK_UPDATES];
    u32 chunk_update_n;
    u32 chunkgen_index;

    VoxelUniformityChunk uniformity_chunks[CHUNK_N];
    ChunkGenState chunks_genstate[CHUNK_N];

#if USING_BRICKMAP
    VoxelChunk generation_chunk;
    VoxelBrick voxel_bricks[BRICKMAP_MAX_N];

    u32 chunk_is_ptr[CHUNK_PTR_U32_N];
    PackedVoxel voxel_chunks[CHUNK_N];
#else
    VoxelChunk voxel_chunks[CHUNK_N];
#endif
};

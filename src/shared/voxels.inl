#pragma once

#include <shared/shapes.inl>

#define CHUNK_SIZE 64
#define VOXEL_SCL 8

#define WORLD_CHUNK_NX 19
#define WORLD_CHUNK_NY 19
#define WORLD_CHUNK_NZ 5
#define WORLD_CHUNK_N (WORLD_CHUNK_NX * WORLD_CHUNK_NY * WORLD_CHUNK_NZ)

#define WORLD_BLOCK_NX (WORLD_CHUNK_NX * CHUNK_SIZE)
#define WORLD_BLOCK_NY (WORLD_CHUNK_NY * CHUNK_SIZE)
#define WORLD_BLOCK_NZ (WORLD_CHUNK_NZ * CHUNK_SIZE)
#define WORLD_BLOCK_N (WORLD_CHUNK_N * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE)

#define BRUSH_CHUNK_NX 4
#define BRUSH_CHUNK_NY 4
#define BRUSH_CHUNK_NZ 4
#define BRUSH_CHUNK_N (BRUSH_CHUNK_NX * BRUSH_CHUNK_NY * BRUSH_CHUNK_NZ)

#define BRUSH_BLOCK_NX (BRUSH_CHUNK_NX * CHUNK_SIZE)
#define BRUSH_BLOCK_NY (BRUSH_CHUNK_NY * CHUNK_SIZE)
#define BRUSH_BLOCK_NZ (BRUSH_CHUNK_NZ * CHUNK_SIZE)
#define BRUSH_BLOCK_N (BRUSH_CHUNK_N * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE)

#define CHUNK_N (WORLD_CHUNK_N + BRUSH_CHUNK_N)

#define MAX_CHUNK_UPDATES ((BRUSH_CHUNK_NX + 1) * (BRUSH_CHUNK_NY + 1) * (BRUSH_CHUNK_NZ + 1))

const u32 CHUNK_PTR_U32_N = (CHUNK_N / 32 == 0) ? 1 : CHUNK_N / 32;

struct Voxel {
    f32vec3 col;
    // f32vec3 nrm;
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

DAXA_DECL_BUFFER_STRUCT(VoxelWorld, {
    Box box;
    f32vec3 center_pt;
    f32 last_update_time;
    u32 chunk_update_indices[MAX_CHUNK_UPDATES];
    u32 chunk_update_n;
    u32 chunkgen_index;
    u32 brush_chunkgen_index;

    VoxelUniformityChunk uniformity_chunks[CHUNK_N];
    ChunkGenState chunks_genstate[CHUNK_N];

    VoxelChunk voxel_chunks[CHUNK_N];
});

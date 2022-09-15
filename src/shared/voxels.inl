#pragma once

#include <shared/shapes.inl>

#define CHUNK_SIZE 64
#define VOXEL_SCL 8

#define CHUNK_NX 1
#define CHUNK_NY 1
#define CHUNK_NZ 1
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

struct VoxelChunk {
    Box box;
    PackedVoxel packed_voxels[CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE];
};
struct VoxelWorld {
    Box box;
    i32vec3 chunkgen_i;
    f32vec3 center_pt;
    VoxelChunk voxel_chunks[CHUNK_N];
};

u32 get_chunk_index(i32vec3 chunk_i) {
    return chunk_i.x + chunk_i.y * CHUNK_NX + chunk_i.z * CHUNK_NX * CHUNK_NY;
}

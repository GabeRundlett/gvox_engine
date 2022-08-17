#pragma once

#include "common/interface/raytrace.hlsl"
#include "common/interface/worldgen.hlsl"

#define CHUNK_SIZE 64
#define VOXEL_SCL 8

uint calc_index(int3 block_offset);

struct EditInfo {
    BlockID block_id;
    float3 col;
    float3 pos;
    float radius;

    void default_init() {
        block_id = BlockID::Air;
        col = 0;
        pos = 0;
        radius = 0;
    }
};

struct Voxel {
    uint col_id;
    uint nrm;

    float3 get_col();
    float3 get_nrm();
};

struct VoxelChunk {
    Voxel data[CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE];
    Box box;

    void default_init() {
        // don't waste time initializing `data`
        box.default_init();
    }

    int3 calc_tile_offset(float3 p);
    uint sample_tile(float3 p);
    uint sample_tile(int3 tile_offset);
    uint sample_tile(uint index);
    Voxel sample_voxel(uint index);
    float3 sample_color(float3 p);
    void gen(int3 block_offset);
    void do_edit(int3 block_offset, in out EditInfo edit_info);
};

template <uint N>
uint uniformity_lod_index(uint3 index_within_lod) {
    return index_within_lod.x + index_within_lod.y * uint(64 / N);
}

uint uniformity_lod_mask(uint3 index_within_lod) {
    return 1u << index_within_lod.z;
}

struct VoxelUniformityChunk {
    uint lod_x2[1024];
    uint lod_x4[256];
    uint lod_x8[64];
    uint lod_x16[16];
    uint lod_x32[4];

    void default_init() {
        // don't waste time initializing `data`
    }

    template <uint N>
    bool lod_nonuniform(uint index, uint mask) {
        return true;
    }
};

#define VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(N)                         \
    template <>                                                           \
    bool VoxelUniformityChunk::lod_nonuniform<N>(uint index, uint mask) { \
        return (lod_x##N[index] & mask) != 0;                             \
    }

VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(2)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(4)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(8)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(16)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(32)

template <>
bool VoxelUniformityChunk::lod_nonuniform<64>(uint index, uint mask) {
    return lod_x32[0] != 0 | lod_x32[1] != 0 | lod_x32[2] != 0 | lod_x32[3] != 0;
}

struct OctreeNode {
    uint ptr;
    uint metadata;
};

struct Octant {
    OctreeNode nodes[8];
};

struct VoxelOctree {
    Octant base;
    Octant data[32 * 32 * 32 + 16 * 16 * 16 + 8 * 8 * 8 + 4 * 4 * 4 + 2 * 2 * 2];
};

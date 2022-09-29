#pragma once

#include <shared/shared.inl>
#include <utils/bitpacking.glsl>

#define BlockID_Debug 0
#define BlockID_Air 1
#define BlockID_Stone 2
#define BlockID_Water 3

#define BlockID_Dirt 4
#define BlockID_Grass 5
#define BlockID_TallGrass 6
#define BlockID_Sand 7
#define BlockID_Sandstone 8
#define BlockID_Gravel 9

#define BlockID_Log 10
#define BlockID_Leaves 11

PackedVoxel pack_voxel(in Voxel voxel) {
    PackedVoxel result;
    result.data = 0;
    result.data = bitpacking_pack_f32vec3(0, 4, result.data, voxel.col);
    result.data = bitpacking_pack_f32vec3(12, 4, result.data, bitpacking_map_range(f32vec3(-1), f32vec3(1), voxel.nrm));
    result.data = bitpacking_pack_u32(24, 8, result.data, voxel.block_id);
    return result;
}
Voxel unpack_voxel(in PackedVoxel packed_voxel) {
    Voxel result;
    result.col = bitpacking_unpack_f32vec3(0, 4, packed_voxel.data);
    result.nrm = bitpacking_unmap_range(f32vec3(-1), f32vec3(1), bitpacking_unpack_f32vec3(12, 4, packed_voxel.data));
    result.block_id = bitpacking_unpack_u32(24, 8, packed_voxel.data);
    return result;
}

u32 get_chunk_index(i32vec3 chunk_i) { return chunk_i.x + chunk_i.y * CHUNK_NX + chunk_i.z * CHUNK_NX * CHUNK_NY; }
f32vec3 get_world_relative_p(f32vec3 p) { return p - VOXEL_WORLD.box.bound_min; }
i32vec3 get_chunk_i(i32vec3 voxel_i) { return voxel_i / CHUNK_SIZE; }
u32vec3 get_inchunk_voxel_i(i32vec3 voxel_i, i32vec3 chunk_i) { return u32vec3(voxel_i - chunk_i * CHUNK_SIZE); }
u32 get_voxel_index(u32vec3 inchunk_voxel_i) { return inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE; }
i32vec3 get_voxel_i(f32vec3 p) { return clamp(i32vec3(get_world_relative_p(p) * VOXEL_SCL), i32vec3(0, 0, 0), i32vec3(BLOCK_NX - 1, BLOCK_NY - 1, BLOCK_NZ - 1)); }
u32vec3 get_subbrick_i0(u32vec3 inchunk_voxel_i) { return inchunk_voxel_i / BRICKMAP_SIZE; }

struct HasSubbricksInfo {
    u32 index;
    u32 mask;
};
HasSubbricksInfo get_has_subbricks_info(u32vec3 subbrick_i) {
    u32 subbrick_index = subbrick_i.x + subbrick_i.y * BRICKMAP_SIZE + subbrick_i.z * BRICKMAP_SIZE * BRICKMAP_SIZE;

    HasSubbricksInfo result;
    result.index = subbrick_index / 32;
    result.mask = 1 << (subbrick_index - result.index * 32);
    return result;
}

struct VoxelWorldSampleInfo {
    i32vec3 voxel_i;
    i32vec3 chunk_i;
    u32vec3 inchunk_voxel_i;
    u32 chunk_index;
    u32 voxel_index;

#if USING_BRICKMAP
    u32 chunk_is_ptr_index;
    u32 chunk_is_ptr_mask;
    u32vec3 subbrick_i0;
    HasSubbricksInfo has_subbricks0;
#endif
};
VoxelWorldSampleInfo get_voxel_world_sample_info(f32vec3 p) {
    VoxelWorldSampleInfo result;
    result.voxel_i = get_voxel_i(p);
    result.chunk_i = get_chunk_i(result.voxel_i);
    result.inchunk_voxel_i = get_inchunk_voxel_i(result.voxel_i, result.chunk_i);
    result.chunk_index = get_chunk_index(result.chunk_i);
    result.voxel_index = get_voxel_index(result.inchunk_voxel_i);

#if USING_BRICKMAP
    result.chunk_is_ptr_index = result.chunk_index / 32;
    result.chunk_is_ptr_mask = 1 << (result.chunk_index - result.chunk_is_ptr_index * 32);
    result.subbrick_i0 = get_subbrick_i0(result.inchunk_voxel_i);
    result.has_subbricks0 = get_has_subbricks_info(result.subbrick_i0);
#endif

    return result;
}

PackedVoxel sample_packed_voxel(u32 chunk_index, u32 voxel_index) {
#if USING_BRICKMAP
    return VOXEL_WORLD.generation_chunk.packed_voxels[voxel_index];
#else
    return VOXEL_CHUNKS[chunk_index].packed_voxels[voxel_index];
#endif
}
PackedVoxel sample_packed_voxel(u32 chunk_index, u32vec3 inchunk_voxel_i) { return sample_packed_voxel(chunk_index, get_voxel_index(inchunk_voxel_i)); }
PackedVoxel sample_packed_voxel(f32vec3 p) {
    i32vec3 voxel_i = get_voxel_i(p);
    i32vec3 chunk_i = get_chunk_i(voxel_i);
    return sample_packed_voxel(get_chunk_index(chunk_i), get_inchunk_voxel_i(voxel_i, chunk_i));
}
u32 sample_voxel_id(u32 chunk_index, u32 voxel_index) { return unpack_voxel(sample_packed_voxel(chunk_index, voxel_index)).block_id; }
u32 sample_voxel_id(u32 chunk_index, u32vec3 inchunk_voxel_i) { return sample_voxel_id(chunk_index, get_voxel_index(inchunk_voxel_i)); }
u32 sample_voxel_id(f32vec3 p) {
    i32vec3 voxel_i = get_voxel_i(p);
    i32vec3 chunk_i = get_chunk_i(voxel_i);
    return sample_voxel_id(get_chunk_index(chunk_i), get_inchunk_voxel_i(voxel_i, chunk_i));
}

#define VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(N)                                       \
    b32 VoxelUniformityChunk_lod_nonuniform_##N(u32 chunk_index, u32 index, u32 mask) { \
        return (UNIFORMITY_CHUNKS[chunk_index].lod_x##N[index] & mask) != 0;            \
    }

VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(2)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(4)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(8)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(16)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(32)

b32 VoxelUniformityChunk_lod_nonuniform_64(u32 chunk_index, u32 index, u32 mask) {
    return (u32(UNIFORMITY_CHUNKS[chunk_index].lod_x32[0] != 0) |
            u32(UNIFORMITY_CHUNKS[chunk_index].lod_x32[1] != 0) |
            u32(UNIFORMITY_CHUNKS[chunk_index].lod_x32[2] != 0) |
            u32(UNIFORMITY_CHUNKS[chunk_index].lod_x32[3] != 0)) != 0;
}

#define voxel_uniformity_lod_nonuniform(N) VoxelUniformityChunk_lod_nonuniform_##N

#define UNIFORMITY_LOD_INDEX_IMPL(N)                                  \
    u32 uniformity_lod_index_##N(u32vec3 index_within_lod) {          \
        return index_within_lod.x + index_within_lod.y * u32(64 / N); \
    }

UNIFORMITY_LOD_INDEX_IMPL(1)
UNIFORMITY_LOD_INDEX_IMPL(2)
UNIFORMITY_LOD_INDEX_IMPL(4)
UNIFORMITY_LOD_INDEX_IMPL(8)
UNIFORMITY_LOD_INDEX_IMPL(16)
UNIFORMITY_LOD_INDEX_IMPL(32)
UNIFORMITY_LOD_INDEX_IMPL(64)

#define uniformity_lod_index(N) uniformity_lod_index_##N

u32 uniformity_lod_mask(u32vec3 index_within_lod) {
    return 1u << index_within_lod.z;
}

#if USING_BRICKMAP

u32 brickmap_depth(in f32vec3 p) {
    VoxelWorldSampleInfo sample_info = get_voxel_world_sample_info(p);

    if ((VOXEL_WORLD.chunk_is_ptr[sample_info.chunk_is_ptr_index] & sample_info.chunk_is_ptr_mask) == 0) {
        return 1;
    }

    u32 brick_index = VOXEL_WORLD.voxel_chunks[sample_info.chunk_index].data;
    u32 has_subbricks = VOXEL_WORLD.voxel_bricks[brick_index].has_subbricks[sample_info.has_subbricks0.index] & sample_info.has_subbricks0.mask;
    has_subbricks = u32(has_subbricks != 0);

    return has_subbricks;
}

#endif

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
#define BlockID_Cobblestone 10

#define BlockID_Log 11
#define BlockID_Leaves 12
#define BlockID_Snow 13

f32vec3 block_color(u32 block_id) {
    // clang-format off
    switch (block_id) {
    case BlockID_Debug:           return f32vec3(0.30, 0.02, 0.30); break;
    case BlockID_Air:             return f32vec3(0.16, 0.17, 0.61); break;
    // case BlockID_Bedrock:         return f32vec3(0.30, 0.30, 0.30); break;
    // case BlockID_Brick:           return f32vec3(0.47, 0.23, 0.20); break;
    // case BlockID_Cactus:          return f32vec3(0.36, 0.62, 0.28); break;
    case BlockID_Cobblestone:     return f32vec3(0.32, 0.31, 0.31); break;
    // case BlockID_CompressedStone: return f32vec3(0.32, 0.31, 0.31); break;
    // case BlockID_DiamondOre:      return f32vec3(0.18, 0.67, 0.69); break;
    case BlockID_Dirt:            return f32vec3(0.08, 0.05, 0.03); break;
    // case BlockID_DriedShrub:      return f32vec3(0.52, 0.36, 0.27); break;
    case BlockID_Grass:           return f32vec3(0.05, 0.09, 0.03); break;
    case BlockID_Gravel:          return f32vec3(0.10, 0.08, 0.07); break;
    // case BlockID_Lava:            return f32vec3(0.00, 0.00, 0.00); break;
    case BlockID_Leaves:          return f32vec3(0.10, 0.29, 0.10) * 0.08; break;
    case BlockID_Log:             return f32vec3(0.06, 0.03, 0.02) * 0.5; break;
    // case BlockID_MoltenRock:      return f32vec3(0.20, 0.13, 0.12); break;
    // case BlockID_Planks:          return f32vec3(0.68, 0.47, 0.35); break;
    // case BlockID_Rose:            return f32vec3(0.52, 0.04, 0.05); break;
    case BlockID_Sand:            return f32vec3(0.33, 0.31, 0.29); break;
    case BlockID_Sandstone:       return f32vec3(0.44, 0.42, 0.41); break;
    case BlockID_Stone:           return f32vec3(0.11, 0.10, 0.09); break;
    case BlockID_TallGrass:       return f32vec3(0.04, 0.08, 0.03); break;
    case BlockID_Water:           return f32vec3(0.20, 0.28, 0.93); break;
    case BlockID_Snow:            return f32vec3(0.60, 0.60, 0.60); break;
    default:                      return f32vec3(0.00, 0.00, 0.00); break;
    }
    // clang-format on
}

PackedVoxel pack_voxel(in Voxel voxel) {
    PackedVoxel result;
    result.data = 0;
    result.data = bitpacking_pack_f32vec3(0, 8, result.data, voxel.col);
    // result.data = bitpacking_pack_f32vec3(12, 4, result.data, bitpacking_map_range(f32vec3(-1), f32vec3(1), voxel.nrm));
    result.data = bitpacking_pack_u32(24, 8, result.data, voxel.block_id);
    return result;
}
Voxel unpack_voxel(in PackedVoxel packed_voxel) {
    Voxel result;
    result.col = bitpacking_unpack_f32vec3(0, 8, packed_voxel.data);
    // result.nrm = bitpacking_unmap_range(f32vec3(-1), f32vec3(1), bitpacking_unpack_f32vec3(12, 4, packed_voxel.data));
    result.block_id = bitpacking_unpack_u32(24, 8, packed_voxel.data);
    return result;
}

struct VoxelSampleInfo {
    i32vec3 voxel_i;
    i32vec3 chunk_i;
    u32vec3 inchunk_voxel_i;
    u32 chunk_index;
    u32 voxel_index;
};

#define impl_get_chunk_index(NAME, ADD) \
    u32 get_chunk_index_##NAME(i32vec3 chunk_i) { return ADD + chunk_i.x + chunk_i.y * NAME##_CHUNK_NX + chunk_i.z * NAME##_CHUNK_NX * NAME##_CHUNK_NY; }
#define impl_get_voxel_i(NAME) \
    i32vec3 get_voxel_i_##NAME(f32vec3 p) { return clamp(i32vec3(p * VOXEL_SCL), i32vec3(0, 0, 0), i32vec3(NAME##_BLOCK_NX - 1, NAME##_BLOCK_NY - 1, NAME##_BLOCK_NZ - 1)); }
#define impl_get_voxel_sample_info(NAME)                                              \
    VoxelSampleInfo get_voxel_sample_info_##NAME(f32vec3 p) {                         \
        VoxelSampleInfo result;                                                       \
        result.voxel_i = get_voxel_i_##NAME(p);                                       \
        result.chunk_i = get_chunk_i(result.voxel_i);                                 \
        result.inchunk_voxel_i = get_inchunk_voxel_i(result.voxel_i, result.chunk_i); \
        result.chunk_index = get_chunk_index_##NAME(result.chunk_i);                  \
        result.voxel_index = get_voxel_index(result.inchunk_voxel_i);                 \
        return result;                                                                \
    }

// clang-format off
i32vec3 get_chunk_i(i32vec3 voxel_i) { return voxel_i / CHUNK_SIZE; }
u32vec3 get_inchunk_voxel_i(i32vec3 voxel_i, i32vec3 chunk_i) { return u32vec3(voxel_i - chunk_i * CHUNK_SIZE); }
u32 get_voxel_index(u32vec3 inchunk_voxel_i) { return inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE; }

impl_get_chunk_index(WORLD, 0)
impl_get_voxel_i(WORLD)
impl_get_voxel_sample_info(WORLD)

impl_get_chunk_index(BRUSH, WORLD_CHUNK_N)
impl_get_voxel_i(BRUSH)
impl_get_voxel_sample_info(BRUSH)

PackedVoxel sample_packed_voxel(u32 chunk_index, u32 voxel_index) {
    return VOXEL_WORLD.voxel_chunks[chunk_index].packed_voxels[voxel_index];
}
PackedVoxel sample_packed_voxel(u32 chunk_index, u32vec3 inchunk_voxel_i) { return sample_packed_voxel(chunk_index, get_voxel_index(inchunk_voxel_i)); }
PackedVoxel sample_packed_voxel(f32vec3 p) {
    i32vec3 voxel_i = get_voxel_i_WORLD(p);
    i32vec3 chunk_i = get_chunk_i(voxel_i);
    return sample_packed_voxel(get_chunk_index_WORLD(chunk_i), get_inchunk_voxel_i(voxel_i, chunk_i));
}
u32 sample_voxel_id(u32 chunk_index, u32 voxel_index) { return unpack_voxel(sample_packed_voxel(chunk_index, voxel_index)).block_id; }
u32 sample_voxel_id(u32 chunk_index, u32vec3 inchunk_voxel_i) { return sample_voxel_id(chunk_index, get_voxel_index(inchunk_voxel_i)); }
u32 sample_voxel_id(f32vec3 p) {
    i32vec3 voxel_i = get_voxel_i_WORLD(p);
    i32vec3 chunk_i = get_chunk_i(voxel_i);
    return sample_voxel_id(get_chunk_index_WORLD(chunk_i), get_inchunk_voxel_i(voxel_i, chunk_i));
}
// clang-format on

#define VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(N)                                        \
    b32 VoxelUniformityChunk_lod_nonuniform_##N(u32 chunk_index, u32 index, u32 mask) {  \
        return (VOXEL_WORLD.uniformity_chunks[chunk_index].lod_x##N[index] & mask) != 0; \
    }

VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(2)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(4)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(8)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(16)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(32)

b32 VoxelUniformityChunk_lod_nonuniform_64(u32 chunk_index, u32 index, u32 mask) {
    return (u32(VOXEL_WORLD.uniformity_chunks[chunk_index].lod_x32[0] != 0) |
            u32(VOXEL_WORLD.uniformity_chunks[chunk_index].lod_x32[1] != 0) |
            u32(VOXEL_WORLD.uniformity_chunks[chunk_index].lod_x32[2] != 0) |
            u32(VOXEL_WORLD.uniformity_chunks[chunk_index].lod_x32[3] != 0)) != 0;
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

#pragma once

#include <shared/shared.inl>
#include <utils/bitpacking.glsl>

#define VOXEL_SCL 8

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
    case BlockID_Grass:           return f32vec3(0.053, 0.101, 0.026); break;
    case BlockID_Gravel:          return f32vec3(0.10, 0.08, 0.07); break;
    // case BlockID_Lava:            return f32vec3(0.00, 0.00, 0.00); break;
    case BlockID_Leaves:          return f32vec3(0.10, 0.29, 0.10) * 0.08; break;
    case BlockID_Log:             return f32vec3(0.06, 0.03, 0.02) * 0.5; break;
    // case BlockID_MoltenRock:      return f32vec3(0.20, 0.13, 0.12); break;
    // case BlockID_Planks:          return f32vec3(0.68, 0.47, 0.35); break;
    // case BlockID_Rose:            return f32vec3(0.52, 0.04, 0.05); break;
    case BlockID_Sand:            return f32vec3(0.65, 0.58, 0.42); break;
    case BlockID_Sandstone:       return f32vec3(0.44, 0.42, 0.41); break;
    case BlockID_Stone:           return f32vec3(0.11, 0.10, 0.09); break;
    case BlockID_TallGrass:       return f32vec3(0.065, 0.116, 0.035); break;
    case BlockID_Water:           return f32vec3(0.20, 0.28, 0.93); break;
    case BlockID_Snow:            return f32vec3(0.60, 0.60, 0.60); break;
    default:                      return f32vec3(0.00, 0.00, 0.00); break;
    }
    // clang-format on
}

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

i32vec3 get_chunk_i(i32vec3 voxel_i) { return voxel_i / CHUNK_SIZE; }
u32vec3 get_inchunk_voxel_i(i32vec3 voxel_i, i32vec3 chunk_i) { return u32vec3(voxel_i - chunk_i * CHUNK_SIZE); }
u32 get_voxel_index(u32vec3 inchunk_voxel_i) {
#if 1
    return inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE;
#elif 0
#define CACHE_BLOCK_AXIS_SIZE 4
#define CACHE_BLOCK_TOTAL_SIZE (CACHE_BLOCK_AXIS_SIZE * CACHE_BLOCK_AXIS_SIZE * CACHE_BLOCK_AXIS_SIZE)
#define CACHE_BLOCK_AXIS_N (CHUNK_SIZE / CACHE_BLOCK_AXIS_SIZE)
#define CACHE_BLOCK_TOTAL_N (CACHE_BLOCK_AXIS_N * CACHE_BLOCK_AXIS_N * CACHE_BLOCK_AXIS_N)
    u32 result_index = inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE;
    u32 block_index = result_index / CACHE_BLOCK_TOTAL_SIZE;
    u32 block_sub_index = result_index - block_index * CACHE_BLOCK_TOTAL_SIZE;
    u32vec3 block_i;
    block_i.z = block_index / (CACHE_BLOCK_AXIS_N * CACHE_BLOCK_AXIS_N);
    block_i.y = (block_index - block_i.z * (CACHE_BLOCK_AXIS_N * CACHE_BLOCK_AXIS_N)) / CACHE_BLOCK_AXIS_N;
    block_i.x = block_index - block_i.z * (CACHE_BLOCK_AXIS_N * CACHE_BLOCK_AXIS_N) - block_i.y * CACHE_BLOCK_AXIS_N;
    u32vec3 result;
    result.z = block_sub_index / (CACHE_BLOCK_AXIS_SIZE * CACHE_BLOCK_AXIS_SIZE);
    result.y = (block_sub_index - result.z * (CACHE_BLOCK_AXIS_SIZE * CACHE_BLOCK_AXIS_SIZE)) / CACHE_BLOCK_AXIS_SIZE;
    result.x = block_sub_index - result.z * (CACHE_BLOCK_AXIS_SIZE * CACHE_BLOCK_AXIS_SIZE) - result.y * CACHE_BLOCK_AXIS_SIZE;
    result += block_i * CACHE_BLOCK_AXIS_SIZE;
    return result.x + result.y * CHUNK_SIZE + result.z * CHUNK_SIZE * CHUNK_SIZE;
#else
    u32vec3 result = inchunk_voxel_i;
    result = (result | (result << u32vec3(8))) & u32vec3(0x0300F00F);
    result = (result | (result << u32vec3(4))) & u32vec3(0x030C30C3);
    result = (result | (result << u32vec3(2))) & u32vec3(0x09249249);
    return (result.x << 0) |
           (result.y << 1) |
           (result.z << 2);
#endif
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

#include <utils/gvox_palette.glsl>

struct VoxelSampleInfo {
    i32vec3 voxel_i;
    i32vec3 chunk_i;
    u32vec3 inchunk_voxel_i;
    u32 chunk_index;
    u32 voxel_index;
};

#define impl_get_chunk_index(NAME) \
    u32 get_chunk_index_##NAME(i32vec3 chunk_i) { return chunk_i.x + chunk_i.y * NAME##_CHUNK_NX + chunk_i.z * NAME##_CHUNK_NX * NAME##_CHUNK_NY; }
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

#define impl_sample_packed_voxel(NAME)                                                                    \
    PackedVoxel sample_packed_voxel_##NAME(f32vec3 p) {                                                   \
        return sample_gvox_palette_voxel(p);                                                              \
    }                                                                                                     \
    PackedVoxel sample_packed_voxel_##NAME(u32 chunk_index, u32vec3 inchunk_voxel_i) {                    \
        return sample_packed_voxel_##NAME(                                                                \
            VOXEL_##NAME.voxel_chunks[chunk_index].box.bound_min + f32vec3(inchunk_voxel_i) / VOXEL_SCL); \
    }                                                                                                     \
    PackedVoxel sample_packed_voxel_##NAME(u32 chunk_index, u32 voxel_index) {                            \
        return sample_packed_voxel_##NAME(                                                                \
            chunk_index,                                                                                  \
            u32vec3(voxel_index % 64, (voxel_index / 64) % 64, voxel_index / 64 / 64));                   \
    }

#define impl_sample_voxel_id(NAME)                                                                                                                                   \
    u32 sample_voxel_id_##NAME(u32 chunk_index, u32 voxel_index) { return unpack_voxel(sample_packed_voxel_##NAME(chunk_index, voxel_index)).block_id; }             \
    u32 sample_voxel_id_##NAME(u32 chunk_index, u32vec3 inchunk_voxel_i) { return unpack_voxel(sample_packed_voxel_##NAME(chunk_index, inchunk_voxel_i)).block_id; } \
    u32 sample_voxel_id_##NAME(f32vec3 p) { return unpack_voxel(sample_packed_voxel_##NAME(p)).block_id; }

#define VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(NAME, N)                                          \
    b32 VoxelUniformityChunk_lod_nonuniform_##NAME##_##N(u32 chunk_index, u32 index, u32 mask) { \
        return (VOXEL_##NAME.uniformity_chunks[chunk_index].lod_x##N[index] & mask) != 0;        \
    }

#define VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM_64(NAME)                                         \
    b32 VoxelUniformityChunk_lod_nonuniform_##NAME##_64(u32 chunk_index, u32 index, u32 mask) { \
        return (u32(VOXEL_##NAME.uniformity_chunks[chunk_index].lod_x32[0] != 0) |              \
                u32(VOXEL_##NAME.uniformity_chunks[chunk_index].lod_x32[1] != 0) |              \
                u32(VOXEL_##NAME.uniformity_chunks[chunk_index].lod_x32[2] != 0) |              \
                u32(VOXEL_##NAME.uniformity_chunks[chunk_index].lod_x32[3] != 0)) != 0;         \
    }

#define voxel_uniformity_lod_nonuniform(NAME, N) VoxelUniformityChunk_lod_nonuniform_##NAME##_##N

// clang-format off
impl_get_chunk_index(WORLD)
impl_get_voxel_i(WORLD)
impl_get_voxel_sample_info(WORLD)
impl_sample_packed_voxel(WORLD)
impl_sample_voxel_id(WORLD)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(WORLD, 2)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(WORLD, 4)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(WORLD, 8)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(WORLD, 16)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(WORLD, 32)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM_64(WORLD)

impl_get_chunk_index(BRUSH)
impl_get_voxel_i(BRUSH)
impl_get_voxel_sample_info(BRUSH)
impl_sample_packed_voxel(BRUSH)
impl_sample_voxel_id(BRUSH)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(BRUSH, 2)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(BRUSH, 4)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(BRUSH, 8)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(BRUSH, 16)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(BRUSH, 32)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM_64(BRUSH)

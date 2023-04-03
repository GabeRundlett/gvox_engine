#pragma once

#include <shared/shared.inl>

#include <utils/math.glsl>
#include <utils/voxel_malloc.glsl>

#define VOXEL_SCL 16

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
u32 uniformity_lod_mask(u32vec3 index_within_lod) {
    return 1u << index_within_lod.z;
}

#define uniformity_lod_index(N) uniformity_lod_index_##N

#define VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(N)                                                                       \
    b32 VoxelUniformityChunk_lod_nonuniform_##N(daxa_BufferPtr(VoxelChunk) voxel_chunk_ptr, u32 index, u32 mask) {      \
        return (deref(voxel_chunk_ptr).uniformity.lod_x##N[index] & mask) != 0;                                         \
    }                                                                                                                   \
    b32 rw_VoxelUniformityChunk_lod_nonuniform_##N(daxa_RWBufferPtr(VoxelChunk) voxel_chunk_ptr, u32 index, u32 mask) { \
        return (deref(voxel_chunk_ptr).uniformity.lod_x##N[index] & mask) != 0;                                         \
    }

VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(2)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(4)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(8)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(16)
VOXEL_UNIFORMITY_CHUNK_IMPL_NONUNIFORM(32)
b32 VoxelUniformityChunk_lod_nonuniform_64(daxa_BufferPtr(VoxelChunk) voxel_chunk_ptr, u32 index, u32 mask) {
    return (u32(deref(voxel_chunk_ptr).uniformity.lod_x32[0] != 0) |
            u32(deref(voxel_chunk_ptr).uniformity.lod_x32[1] != 0) |
            u32(deref(voxel_chunk_ptr).uniformity.lod_x32[2] != 0) |
            u32(deref(voxel_chunk_ptr).uniformity.lod_x32[3] != 0)) != 0;
}

#define voxel_uniformity_lod_nonuniform(N) VoxelUniformityChunk_lod_nonuniform_##N
#define voxel_rw_uniformity_lod_nonuniform(N) rw_VoxelUniformityChunk_lod_nonuniform_##N

u32 calc_chunk_index(u32vec3 chunk_i, u32vec3 chunk_n) {
    u32 chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
    return chunk_index;
}

#define MODEL deref(model_ptr)
u32 sample_gvox_palette_voxel(daxa_BufferPtr(GpuGvoxModel) model_ptr, u32vec3 voxel_i, u32 channel_index) {
    u32 packed_voxel_data;
    packed_voxel_data = 0;
    u32vec3 model_size = u32vec3(MODEL.extent_x, MODEL.extent_y, MODEL.extent_z);
    u32vec3 region_n = (model_size + PALETTE_REGION_SIZE - 1) / PALETTE_REGION_SIZE;
    // const f32vec3 model_off = f32vec3(MODEL.offset_x, MODEL.offset_y, MODEL.offset_z) / VOXEL_SCL;
    // sample_p = sample_p + model_off;
    // i32vec3 voxel_i = i32vec3(sample_p * VOXEL_SCL) - i32vec3(MODEL.offset_x, MODEL.offset_y, MODEL.offset_z);
    // voxel_i -= i32vec3(sample_p.x < 0, sample_p.y < 0, sample_p.z < 0);
    if (
        // voxel_i.x < 0 || voxel_i.y < 0 || voxel_i.z < 0 ||
        voxel_i.x >= model_size.x || voxel_i.y >= model_size.y || voxel_i.z >= model_size.z) {
        return packed_voxel_data;
    }
    u32 region_header_n = region_n.x * region_n.y * region_n.z;
    u32vec3 region_i = voxel_i / PALETTE_REGION_SIZE;
    u32vec3 in_region_i = voxel_i - region_i * PALETTE_REGION_SIZE;
    u32 region_index = region_i.x + region_i.y * region_n.x + region_i.z * region_n.x * region_n.y;
    u32 in_region_index = in_region_i.x + in_region_i.y * PALETTE_REGION_SIZE + in_region_i.z * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE;
    u32 channel_offset = (region_index * MODEL.channel_n + channel_index) * 2;
    u32 variant_n = MODEL.data[channel_offset + 0];
    u32 blob_ptr = MODEL.data[channel_offset + 1];
    u32 v_data_offset = 2 * region_header_n * MODEL.channel_n + blob_ptr / 4;
    u32 bits_per_variant = ceil_log2(variant_n);
    if (variant_n > PALETTE_MAX_COMPRESSED_VARIANT_N) {
        packed_voxel_data = MODEL.data[v_data_offset + in_region_index];
    } else if (variant_n > 1) {
        u32 mask = (~0u) >> (32 - bits_per_variant);
        u32 bit_index = in_region_index * bits_per_variant;
        u32 data_index = bit_index / 32;
        u32 data_offset = bit_index - data_index * 32;
        u32 my_palette_index = (MODEL.data[v_data_offset + variant_n + data_index + 0] >> data_offset) & mask;
        if (data_offset + bits_per_variant > 32) {
            u32 shift = bits_per_variant - ((data_offset + bits_per_variant) & 0x1f);
            my_palette_index |= (MODEL.data[v_data_offset + variant_n + data_index + 1] << shift) & mask;
        }
        packed_voxel_data = MODEL.data[v_data_offset + my_palette_index];
    } else {
        packed_voxel_data = blob_ptr;
    }
    return packed_voxel_data;
}
#undef MODEL

#define READ_FROM_HEAP 1
u32 sample_voxel_chunk(daxa_RWBufferPtr(VoxelMalloc_GlobalAllocator) allocator, daxa_BufferPtr(VoxelChunk) voxel_chunk_ptr, u32vec3 inchunk_voxel_i, bool is_drawing) {
    u32vec3 palette_region_i = inchunk_voxel_i / PALETTE_REGION_SIZE;
    u32vec3 palette_voxel_i = inchunk_voxel_i - palette_region_i * PALETTE_REGION_SIZE;
    u32 palette_region_index = palette_region_i.x + palette_region_i.y * PALETTES_PER_CHUNK_AXIS + palette_region_i.z * PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS;
    u32 palette_voxel_index = palette_voxel_i.x + palette_voxel_i.y * PALETTE_REGION_SIZE + palette_voxel_i.z * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE;
    PaletteHeader palette_header = deref(voxel_chunk_ptr).palette_headers[palette_region_index];
    if (palette_header.variant_n == 1) {
        return palette_header.blob_ptr;
    }
#if READ_FROM_HEAP
    daxa_RWBufferPtr(daxa_u32) blob_u32s = voxel_malloc_address_to_u32_ptr(allocator, palette_header.blob_ptr);
#endif
    if (palette_header.variant_n > PALETTE_MAX_COMPRESSED_VARIANT_N) {
#if READ_FROM_HEAP
        return deref(blob_u32s[palette_voxel_index]);
#else
        return 0x01ffff00;
#endif
    }
#if READ_FROM_HEAP
    u32 bits_per_variant = ceil_log2(palette_header.variant_n);
    u32 mask = (~0u) >> (32 - bits_per_variant);
    u32 bit_index = palette_voxel_index * bits_per_variant;
    u32 data_index = bit_index / 32;
    u32 data_offset = bit_index - data_index * 32;
    u32 my_palette_index = (deref(blob_u32s[palette_header.variant_n + data_index + 0]) >> data_offset) & mask;
    if (data_offset + bits_per_variant > 32) {
        u32 shift = bits_per_variant - ((data_offset + bits_per_variant) & 0x1f);
        my_palette_index |= (deref(blob_u32s[palette_header.variant_n + data_index + 1]) << shift) & mask;
    }
    u32 voxel_data = deref(blob_u32s[my_palette_index]);
    if (is_drawing) {
        return voxel_data;
        // return ((palette_header.blob_ptr / 512) & 0x00ffffff) | (voxel_data & 0xff000000);
    } else {
        return voxel_data;
    }
#else
    return 0x01ff00ff;
#endif
}

u32 sample_voxel_chunk(daxa_RWBufferPtr(VoxelMalloc_GlobalAllocator) allocator, daxa_BufferPtr(VoxelChunk) voxel_chunks_ptr, u32vec3 chunk_n, u32vec3 voxel_i, bool is_drawing) {
    u32vec3 chunk_i = voxel_i / CHUNK_SIZE;
    u32vec3 inchunk_voxel_i = voxel_i - chunk_i * CHUNK_SIZE;
    u32 chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
    if (chunk_i.x >= chunk_n.x || chunk_i.y >= chunk_n.y || chunk_i.z >= chunk_n.z) {
        return 0u;
    }
    daxa_BufferPtr(VoxelChunk) voxel_chunk_ptr = voxel_chunks_ptr[chunk_index];
    return sample_voxel_chunk(allocator, voxel_chunk_ptr, inchunk_voxel_i, is_drawing);
}

#define SAMPLE_LOD_PRESENCE_IMPL(N)                                                                              \
    b32 sample_lod_presence_##N(daxa_BufferPtr(VoxelChunk) voxel_chunks_ptr, u32vec3 chunk_n, u32vec3 voxel_i) { \
        u32vec3 chunk_i = voxel_i / CHUNK_SIZE;                                                                  \
        u32vec3 inchunk_voxel_i = voxel_i - chunk_i * CHUNK_SIZE;                                                \
        u32 chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;                 \
        daxa_BufferPtr(VoxelChunk) voxel_chunk_ptr = voxel_chunks_ptr[chunk_index];                              \
        u32 lod_index = uniformity_lod_index(N)(inchunk_voxel_i / N);                                            \
        u32 lod_mask = uniformity_lod_mask(inchunk_voxel_i / N);                                                 \
        return voxel_uniformity_lod_nonuniform(N)(voxel_chunk_ptr, lod_index, lod_mask);                         \
    }
SAMPLE_LOD_PRESENCE_IMPL(2)
SAMPLE_LOD_PRESENCE_IMPL(4)
SAMPLE_LOD_PRESENCE_IMPL(8)
SAMPLE_LOD_PRESENCE_IMPL(16)
SAMPLE_LOD_PRESENCE_IMPL(32)
SAMPLE_LOD_PRESENCE_IMPL(64)

#define sample_lod_presence(N) sample_lod_presence_##N

u32 sample_lod(daxa_RWBufferPtr(VoxelMalloc_GlobalAllocator) allocator, daxa_BufferPtr(VoxelChunk) voxel_chunk_ptr, u32vec3 chunk_i, u32vec3 inchunk_voxel_i) {
    u32 lod_index_x2 = uniformity_lod_index(2)(inchunk_voxel_i / 2);
    u32 lod_mask_x2 = uniformity_lod_mask(inchunk_voxel_i / 2);
    u32 lod_index_x4 = uniformity_lod_index(4)(inchunk_voxel_i / 4);
    u32 lod_mask_x4 = uniformity_lod_mask(inchunk_voxel_i / 4);
    u32 lod_index_x8 = uniformity_lod_index(8)(inchunk_voxel_i / 8);
    u32 lod_mask_x8 = uniformity_lod_mask(inchunk_voxel_i / 8);
    u32 lod_index_x16 = uniformity_lod_index(16)(inchunk_voxel_i / 16);
    u32 lod_mask_x16 = uniformity_lod_mask(inchunk_voxel_i / 16);
    u32 lod_index_x32 = uniformity_lod_index(32)(inchunk_voxel_i / 32);
    u32 lod_mask_x32 = uniformity_lod_mask(inchunk_voxel_i / 32);
    u32 lod_index_x64 = uniformity_lod_index(64)(inchunk_voxel_i / 64);
    u32 lod_mask_x64 = uniformity_lod_mask(inchunk_voxel_i / 64);

    u32 chunk_edit_stage = deref(voxel_chunk_ptr).edit_stage;
    if (chunk_edit_stage == 0)
        return 7;
    if ((sample_voxel_chunk(allocator, voxel_chunk_ptr, inchunk_voxel_i, false) & 0xff000000) != 0)
        return 0;
    if (voxel_uniformity_lod_nonuniform(2)(voxel_chunk_ptr, lod_index_x2, lod_mask_x2))
        return 1;
    if (voxel_uniformity_lod_nonuniform(4)(voxel_chunk_ptr, lod_index_x4, lod_mask_x4))
        return 2;
    if (voxel_uniformity_lod_nonuniform(8)(voxel_chunk_ptr, lod_index_x8, lod_mask_x8))
        return 3;
    if (voxel_uniformity_lod_nonuniform(16)(voxel_chunk_ptr, lod_index_x16, lod_mask_x16))
        return 4;
    if (voxel_uniformity_lod_nonuniform(32)(voxel_chunk_ptr, lod_index_x32, lod_mask_x32))
        return 5;
    if (voxel_uniformity_lod_nonuniform(64)(voxel_chunk_ptr, lod_index_x64, lod_mask_x64))
        return 6;

    return 7;
}

u32 sample_lod(daxa_RWBufferPtr(VoxelMalloc_GlobalAllocator) allocator, daxa_BufferPtr(VoxelChunk) voxel_chunks_ptr, u32vec3 chunk_n, f32vec3 voxel_p) {
    // u32vec3 voxel_i = u32vec3(clamp(voxel_p * VOXEL_SCL, f32vec3(0, 0, 0), (f32vec3(chunk_n) * CHUNK_SIZE - 1) / VOXEL_SCL));
    u32vec3 voxel_i = u32vec3(voxel_p * VOXEL_SCL);
    u32vec3 chunk_i = voxel_i / CHUNK_SIZE;
    u32 chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
    return sample_lod(allocator, voxel_chunks_ptr[chunk_index], chunk_i, voxel_i - chunk_i * CHUNK_SIZE);
}

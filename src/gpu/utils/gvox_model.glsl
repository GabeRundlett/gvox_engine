#pragma once

#include <shared/shared.inl>

#define VOXEL_SCL 8
#define PALETTE_REGION_SIZE 8

#if PALETTE_REGION_SIZE == 8
#define PALETTE_MAX_COMPRESSED_VARIANT_N 367
#elif PALETTE_REGION_SIZE == 16
#define PALETTE_MAX_COMPRESSED_VARIANT_N 2559
#else
#error Unsupported Palette Region Size
#endif

u32 ceil_log2(u32 x) {
    u32 t[5] = u32[5](
        0xFFFF0000,
        0x0000FF00,
        0x000000F0,
        0x0000000C,
        0x00000002);
    u32 y = (((x & (x - 1)) == 0) ? 0 : 1);
    i32 j = 16;
    for (u32 i = 0; i < 5; i++) {
        i32 k = (((x & t[i]) == 0) ? 0 : j);
        y += u32(k);
        x >>= k;
        j >>= 1;
    }
    return y;
}

#define MODEL deref(model_ptr)
u32 sample_gvox_palette_voxel(daxa_BufferPtr(GpuGvoxModel) model_ptr, f32vec3 sample_p, u32 channel_index) {
    const f32vec3 model_off = f32vec3(MODEL.offset_x, MODEL.offset_y, MODEL.offset_z) / VOXEL_SCL;
    sample_p = sample_p + model_off;
    u32 packed_voxel_data;
    packed_voxel_data = 0;
    u32vec3 region_n = u32vec3(
        (MODEL.extent_x + (PALETTE_REGION_SIZE - 1)) / PALETTE_REGION_SIZE,
        (MODEL.extent_y + (PALETTE_REGION_SIZE - 1)) / PALETTE_REGION_SIZE,
        (MODEL.extent_z + (PALETTE_REGION_SIZE - 1)) / PALETTE_REGION_SIZE);
    u32vec3 model_size = u32vec3(MODEL.extent_x, MODEL.extent_y, MODEL.extent_z);
    u32 region_header_n = region_n.x * region_n.y * region_n.z;
    i32vec3 p = i32vec3(sample_p * VOXEL_SCL) - i32vec3(MODEL.offset_x, MODEL.offset_y, MODEL.offset_z);
    if (p.x < 0 || p.y < 0 || p.z < 0 ||
        p.x >= model_size.x || p.y >= model_size.y || p.z >= model_size.z) {
        return packed_voxel_data;
    }
    u32vec3 region_i = u32vec3(p / PALETTE_REGION_SIZE);
    u32vec3 in_region_i = u32vec3(p) - region_i * PALETTE_REGION_SIZE;
    u32 region_index = region_i.x + region_i.y * region_n.x + region_i.z * region_n.x * region_n.y;
    u32 in_region_index = in_region_i.x + in_region_i.y * PALETTE_REGION_SIZE + in_region_i.z * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE;
    u32 channel_offset = (region_index * MODEL.channel_n + channel_index) * 2;
    u32 variant_n = MODEL.data[channel_offset + 0];
    u32 blob_offset = MODEL.data[channel_offset + 1];
    u32 v_data_offset = 2 * region_header_n * MODEL.channel_n + blob_offset / 4;
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
        packed_voxel_data = blob_offset;
    }
    return packed_voxel_data;
}
#undef MODEL

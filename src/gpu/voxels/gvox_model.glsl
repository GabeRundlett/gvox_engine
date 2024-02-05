#pragma once

#include <shared/voxels/gvox_model.inl>

#define MODEL deref(model_ptr)
daxa_u32 sample_gvox_palette_voxel(daxa_BufferPtr(GpuGvoxModel) model_ptr, daxa_i32vec3 voxel_i, daxa_u32 channel_index) {
    daxa_u32 packed_voxel_data;
    packed_voxel_data = 0;
    daxa_u32vec3 model_size = daxa_u32vec3(MODEL.extent_x, MODEL.extent_y, MODEL.extent_z);
    daxa_u32vec3 region_n = (model_size + PALETTE_REGION_SIZE - 1) / PALETTE_REGION_SIZE;
    // const daxa_f32vec3 model_off = daxa_f32vec3(MODEL.offset_x, MODEL.offset_y, MODEL.offset_z) / VOXEL_SCL;
    // sample_p = sample_p + model_off;
    // daxa_i32vec3 voxel_i = daxa_i32vec3(sample_p * VOXEL_SCL) - daxa_i32vec3(MODEL.offset_x, MODEL.offset_y, MODEL.offset_z);
    // voxel_i -= daxa_i32vec3(sample_p.x < 0, sample_p.y < 0, sample_p.z < 0);
    if (
        voxel_i.x < 0 || voxel_i.y < 0 || voxel_i.z < 0 ||
        voxel_i.x >= model_size.x || voxel_i.y >= model_size.y || voxel_i.z >= model_size.z) {
        return packed_voxel_data;
    }
    daxa_u32 region_header_n = region_n.x * region_n.y * region_n.z;
    daxa_u32vec3 region_i = daxa_u32vec3(voxel_i) / PALETTE_REGION_SIZE;
    daxa_u32vec3 in_region_i = daxa_u32vec3(voxel_i) - region_i * PALETTE_REGION_SIZE;
    daxa_u32 region_index = region_i.x + region_i.y * region_n.x + region_i.z * region_n.x * region_n.y;
    daxa_u32 in_region_index = in_region_i.x + in_region_i.y * PALETTE_REGION_SIZE + in_region_i.z * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE;
    daxa_u32 channel_offset = (region_index * MODEL.channel_n + channel_index) * 2;
    daxa_u32 variant_n = MODEL.data[channel_offset + 0];
    daxa_u32 blob_ptr = MODEL.data[channel_offset + 1];
    daxa_u32 v_data_offset = 2 * region_header_n * MODEL.channel_n + blob_ptr / 4;
    daxa_u32 bits_per_variant = ceil_log2(variant_n);
    if (variant_n > PALETTE_MAX_COMPRESSED_VARIANT_N) {
        packed_voxel_data = MODEL.data[v_data_offset + in_region_index];
    } else if (variant_n > 1) {
        daxa_u32 mask = (~0u) >> (32 - bits_per_variant);
        daxa_u32 bit_index = in_region_index * bits_per_variant;
        daxa_u32 data_index = bit_index / 32;
        daxa_u32 data_offset = bit_index - data_index * 32;
        daxa_u32 my_palette_index = (MODEL.data[v_data_offset + variant_n + data_index + 0] >> data_offset) & mask;
        if (data_offset + bits_per_variant > 32) {
            daxa_u32 shift = bits_per_variant - ((data_offset + bits_per_variant) & 0x1f);
            my_palette_index |= (MODEL.data[v_data_offset + variant_n + data_index + 1] << shift) & mask;
        }
        packed_voxel_data = MODEL.data[v_data_offset + my_palette_index];
    } else {
        packed_voxel_data = blob_ptr;
    }
    return packed_voxel_data;
}
#undef MODEL

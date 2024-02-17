#pragma once

#include <voxels/gvox_model.inl>

#define MODEL deref(model_ptr)
uint sample_gvox_palette_voxel(daxa_BufferPtr(GpuGvoxModel) model_ptr, ivec3 voxel_i, uint channel_index) {
    uint packed_voxel_data;
    packed_voxel_data = 0;
    uvec3 model_size = uvec3(MODEL.extent_x, MODEL.extent_y, MODEL.extent_z);
    uvec3 region_n = (model_size + PALETTE_REGION_SIZE - 1) / PALETTE_REGION_SIZE;
    // const vec3 model_off = vec3(MODEL.offset_x, MODEL.offset_y, MODEL.offset_z) / VOXEL_SCL;
    // sample_p = sample_p + model_off;
    // ivec3 voxel_i = ivec3(sample_p * VOXEL_SCL) - ivec3(MODEL.offset_x, MODEL.offset_y, MODEL.offset_z);
    // voxel_i -= ivec3(sample_p.x < 0, sample_p.y < 0, sample_p.z < 0);
    if (
        voxel_i.x < 0 || voxel_i.y < 0 || voxel_i.z < 0 ||
        voxel_i.x >= model_size.x || voxel_i.y >= model_size.y || voxel_i.z >= model_size.z) {
        return packed_voxel_data;
    }
    uint region_header_n = region_n.x * region_n.y * region_n.z;
    uvec3 region_i = uvec3(voxel_i) / PALETTE_REGION_SIZE;
    uvec3 in_region_i = uvec3(voxel_i) - region_i * PALETTE_REGION_SIZE;
    uint region_index = region_i.x + region_i.y * region_n.x + region_i.z * region_n.x * region_n.y;
    uint in_region_index = in_region_i.x + in_region_i.y * PALETTE_REGION_SIZE + in_region_i.z * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE;
    uint channel_offset = (region_index * MODEL.channel_n + channel_index) * 2;
    uint variant_n = MODEL.data[channel_offset + 0];
    uint blob_ptr = MODEL.data[channel_offset + 1];
    uint v_data_offset = 2 * region_header_n * MODEL.channel_n + blob_ptr / 4;
    uint bits_per_variant = ceil_log2(variant_n);
    if (variant_n > PALETTE_MAX_COMPRESSED_VARIANT_N) {
        packed_voxel_data = MODEL.data[v_data_offset + in_region_index];
    } else if (variant_n > 1) {
        uint mask = (~0u) >> (32 - bits_per_variant);
        uint bit_index = in_region_index * bits_per_variant;
        uint data_index = bit_index / 32;
        uint data_offset = bit_index - data_index * 32;
        uint my_palette_index = (MODEL.data[v_data_offset + variant_n + data_index + 0] >> data_offset) & mask;
        if (data_offset + bits_per_variant > 32) {
            uint shift = bits_per_variant - ((data_offset + bits_per_variant) & 0x1f);
            my_palette_index |= (MODEL.data[v_data_offset + variant_n + data_index + 1] << shift) & mask;
        }
        packed_voxel_data = MODEL.data[v_data_offset + my_palette_index];
    } else {
        packed_voxel_data = blob_ptr;
    }
    return packed_voxel_data;
}
#undef MODEL

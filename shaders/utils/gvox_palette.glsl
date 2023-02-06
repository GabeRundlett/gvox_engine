#pragma once

u32 ceil_log2(u32 x) {
    u32 t[6] = u32[6](
        0x00000000,
        0xFFFF0000,
        0x0000FF00,
        0x000000F0,
        0x0000000C,
        0x00000002);

    u32 y = (((x & (x - 1)) == 0) ? 0 : 1);
    i32 j = 32;

    for (u32 i = 0; i < 6; i++) {
        i32 k = (((x & t[i]) == 0) ? 0 : j);
        y += u32(k);
        x >>= k;
        j >>= 1;
    }

    return y;
}

PackedVoxel sample_gvox_palette_voxel(f32vec3 sample_p) {
    PackedVoxel packed_voxel;
    packed_voxel.data = 0x01000000;

    u32vec3 region_n = u32vec3(MODEL.node_header.region_count_x, MODEL.node_header.region_count_y, MODEL.node_header.region_count_z);
    u32vec3 model_size = region_n * 8;
    u32 region_header_n = region_n.x * region_n.y * region_n.z;

    i32vec3 p = i32vec3(sample_p * VOXEL_SCL);
    if (p.x < 0 || p.y < 0 || p.z < 0 ||
        p.x >= model_size.x || p.y >= model_size.y || p.z >= model_size.z) {
        return packed_voxel;
    }
    p = p % i32vec3(model_size);

    u32vec3 region_i = u32vec3(p / 8);
    u32vec3 in_region_i = u32vec3(p) - region_i * 8;

    u32 region_index = region_i.x + region_i.y * region_n.x + region_i.z * region_n.x * region_n.y;
    u32 in_region_index = in_region_i.x + in_region_i.y * 8 + in_region_i.z * 8 * 8;

    GpuGVox_RegionHeader region_header;
    region_header.variant_n = MODEL.data[0 + region_index * 2 + 0];
    region_header.blob_offset = MODEL.data[0 + region_index * 2 + 1];

    u32 v_data_offset = 2 * region_header_n + region_header.blob_offset / 4;

    u32 bits_per_variant = ceil_log2(region_header.variant_n);

    // if (region_header.variant_n > 367) {
    // } else
    if (region_header.variant_n > 1) {
        u32 mask = (~0u) >> (32 - bits_per_variant);
        u32 bit_index = in_region_index * bits_per_variant;
        u32 data_index = bit_index / 32;
        u32 data_offset = bit_index - data_index * 32;
        u32 my_palette_index = (MODEL.data[v_data_offset + region_header.variant_n + data_index + 0] >> data_offset) & mask;
        if (data_offset + bits_per_variant > 32) {
            u32 shift = bits_per_variant - ((data_offset + bits_per_variant) & 0x1f);
            my_palette_index |= (MODEL.data[v_data_offset + region_header.variant_n + data_index + 1] << shift) & mask;
        }
        packed_voxel.data = MODEL.data[v_data_offset + my_palette_index];
    } else {
        packed_voxel.data = region_header.blob_offset;
    }

    // Voxel voxel = unpack_voxel(packed_voxel);
    // if (voxel.block_id == 0) {
    //     voxel.block_id = BlockID_Air;
    // } else {
    //     voxel.block_id = BlockID_Stone;
    // }
    // packed_voxel = pack_voxel(voxel);
    if ((packed_voxel.data & 0xff000000) == 0) {
        packed_voxel.data |= 0x01000000;
    } else {
        packed_voxel.data |= 0x02000000;
    }

    return packed_voxel;
}

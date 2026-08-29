#include "common.hpp"

export void generate_bitmask(
    uniform int brick_xi, uniform int brick_yi, uniform int brick_zi,
    uniform int chunk_xi, uniform int chunk_yi, uniform int chunk_zi,
    uniform int level_i, uniform uint bits[], uint *uniform metadata,
    uniform NoiseSettings const *uniform noise_settings, RandomCtx random_ctx) {

    bool has_voxel = false;
    bool has_air_nx = false;
    bool has_air_px = false;
    bool has_air_ny = false;
    bool has_air_py = false;
    bool has_air_nz = false;
    bool has_air_pz = false;

    foreach (word_i = 0 ...(BRICK_SIZE * BRICK_SIZE * BRICK_SIZE / 32)) {
        uint word_result = 0;
#pragma unroll
        for (uniform uint in_word_i = 0; in_word_i < 32; ++in_word_i) {
            uint i = word_i * 32 + in_word_i;
            int xi = i & ((1 << BRICK_SIZE_LOG2) - 1);
            int yi = (i >> BRICK_SIZE_LOG2) & ((1 << BRICK_SIZE_LOG2) - 1);
            int zi = (i >> (BRICK_SIZE_LOG2 * 2)) & ((1 << BRICK_SIZE_LOG2) - 1);

            vec3 pos;
            pos.x = (xi + brick_xi * BRICK_SIZE + chunk_xi * CHUNK_SIZE_VOXELS) * (1 << level_i);
            pos.y = (yi + brick_yi * BRICK_SIZE + chunk_yi * CHUNK_SIZE_VOXELS) * (1 << level_i);
            pos.z = (zi + brick_zi * BRICK_SIZE + chunk_zi * CHUNK_SIZE_VOXELS) * (1 << level_i);

            pos.x = (pos.x + 0.5f) * VOXEL_SIZE;
            pos.y = (pos.y + 0.5f) * VOXEL_SIZE;
            pos.z = (pos.z + 0.5f) * VOXEL_SIZE;

            DensityNrm dn = voxel_value(random_ctx, noise_settings, pos);
            uint value = dn.val < 0.0f;
            word_result |= value << in_word_i;

            has_voxel = has_voxel || (value != 0);
            has_air_nx = has_air_nx || (xi == 0 && value == 0);
            has_air_px = has_air_px || (xi == (BRICK_SIZE - 1) && value == 0);
            has_air_ny = has_air_ny || (yi == 0 && value == 0);
            has_air_py = has_air_py || (yi == (BRICK_SIZE - 1) && value == 0);
            has_air_nz = has_air_nz || (zi == 0 && value == 0);
            has_air_pz = has_air_pz || (zi == (BRICK_SIZE - 1) && value == 0);
        }
        bits[word_i] = word_result;
    }

    if (packmask(has_voxel) != 0) {
        *metadata |= (1 << 22);
    }
    if (packmask(has_air_nx) != 0) {
        *metadata |= (1 << 16);
    }
    if (packmask(has_air_px) != 0) {
        *metadata |= (1 << 19);
    }
    if (packmask(has_air_ny) != 0) {
        *metadata |= (1 << 17);
    }
    if (packmask(has_air_py) != 0) {
        *metadata |= (1 << 20);
    }
    if (packmask(has_air_nz) != 0) {
        *metadata |= (1 << 18);
    }
    if (packmask(has_air_pz) != 0) {
        *metadata |= (1 << 21);
    }
}

export void generate_attributes(
    uniform int brick_xi, uniform int brick_yi, uniform int brick_zi,
    uniform int chunk_xi, uniform int chunk_yi, uniform int chunk_zi,
    uniform int level_i, uniform uint packed_voxels[], uniform uint foliage_bits[],
    uniform NoiseSettings const *uniform noise_settings, RandomCtx random_ctx) {

    foreach (word_i = 0 ...(BRICK_SIZE * BRICK_SIZE * BRICK_SIZE / 32)) {
        uint foliage_word_result = 0;
        for (uniform uint in_word_i = 0; in_word_i < 32; ++in_word_i) {
            uint voxel_index = word_i * 32 + in_word_i;
            int xi = voxel_index & ((1 << BRICK_SIZE_LOG2) - 1);
            int yi = (voxel_index >> BRICK_SIZE_LOG2) & ((1 << BRICK_SIZE_LOG2) - 1);
            int zi = (voxel_index >> (BRICK_SIZE_LOG2 * 2)) & ((1 << BRICK_SIZE_LOG2) - 1);

            vec3 pos;
            pos.x = (xi + brick_xi * BRICK_SIZE + chunk_xi * CHUNK_SIZE_VOXELS) * (1 << level_i);
            pos.y = (yi + brick_yi * BRICK_SIZE + chunk_yi * CHUNK_SIZE_VOXELS) * (1 << level_i);
            pos.z = (zi + brick_zi * BRICK_SIZE + chunk_zi * CHUNK_SIZE_VOXELS) * (1 << level_i);

            pos.x = (pos.x + 0.5f) * VOXEL_SIZE;
            pos.y = (pos.y + 0.5f) * VOXEL_SIZE;
            pos.z = (pos.z + 0.5f) * VOXEL_SIZE;

            vec3 above_pos = pos;
            above_pos.z += VOXEL_SIZE;
            DensityNrm dn = voxel_value(random_ctx, noise_settings, pos);
            DensityNrm dn_above = voxel_value(random_ctx, noise_settings, above_pos);
            uint self_is_solid = dn.val < 0.0f;
            uint above_is_solid = dn_above.val < 0.0f;

            Voxel voxel;
            float upwards = dot(dn.nrm, UP);
            float r = random_ctx[voxel_index] / 255.0f;

            if (dot(dn.nrm, UP) > 0.65f && dn.val > -2.5f) {
                // voxel.albedo = hsv2rgb(vec3(0.18 + 0.005, 0.8, 0.4));
                voxel.albedo = GRASS_COL;
                voxel.roughness = 0.9;
                uint value = above_is_solid == 0 && self_is_solid != 0 && dot(dn.nrm, UP) > 0.65f && r > 0.25f;
                foliage_word_result |= value << in_word_i;
            } else if (dn.val > -0.5 && upwards > 0.40) {
                voxel.albedo = DIRT_COL;
                if (r < 0.5) {
                    voxel.albedo.r *= 0.5;
                    voxel.albedo.g *= 0.5;
                    voxel.albedo.b *= 0.5;
                    voxel.roughness = 1;
                } else if (r < 0.52) {
                    voxel.albedo.r *= 1.5;
                    voxel.albedo.g *= 1.5;
                    voxel.albedo.b *= 1.5;
                    voxel.roughness = 1;
                }
            } else if (dn.val < -0.01 && dn.val > -0.5 && upwards > 0.2) {
                voxel.albedo = GRAVEL_COL;
                if (r < 0.5) {
                    voxel.albedo.r *= 0.75;
                    voxel.albedo.g *= 0.75;
                    voxel.albedo.b *= 0.75;
                }
                voxel.roughness = 1;
            } else {
                voxel.albedo = STONE_COL;
                voxel.roughness = 1;
            }

            ivec3 o = {xi, yi, zi};
            voxel.normal = dither_nrm(random_ctx, dn.nrm, o);
            voxel.material_type = 0;
            PackedVoxel packed_voxel = pack_voxel(voxel);
            packed_voxels[voxel_index] = packed_voxel.data;
            // densities[voxel_index] = dn.val;
        }
        foliage_bits[word_i] = foliage_word_result;
    }
}

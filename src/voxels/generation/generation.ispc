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

    foreach (word_i = 0 ... (VOXEL_BRICK_SIZE * VOXEL_BRICK_SIZE * VOXEL_BRICK_SIZE / 32)) {
        uint word_result = 0;
        #pragma unroll
        for (uniform uint in_word_i = 0; in_word_i < 32; ++in_word_i) {
            uint i = word_i * 32 + in_word_i;
            int xi = i & ((1 << VOXEL_BRICK_SIZE_LOG2) - 1);
            int yi = (i >> VOXEL_BRICK_SIZE_LOG2) & ((1 << VOXEL_BRICK_SIZE_LOG2) - 1);
            int zi = (i >> (VOXEL_BRICK_SIZE_LOG2 * 2)) & ((1 << VOXEL_BRICK_SIZE_LOG2) - 1);

            vec3 pos;
            pos.x = (xi + brick_xi * VOXEL_BRICK_SIZE + chunk_xi * VOXEL_CHUNK_SIZE) * (1 << level_i);
            pos.y = (yi + brick_yi * VOXEL_BRICK_SIZE + chunk_yi * VOXEL_CHUNK_SIZE) * (1 << level_i);
            pos.z = (zi + brick_zi * VOXEL_BRICK_SIZE + chunk_zi * VOXEL_CHUNK_SIZE) * (1 << level_i);

            pos.x = (pos.x + 0.5f) * VOXEL_SIZE;
            pos.y = (pos.y + 0.5f) * VOXEL_SIZE;
            pos.z = (pos.z + 0.5f) * VOXEL_SIZE;

            DensityNrm dn = voxel_value(random_ctx, noise_settings, pos);
            uint value = dn.val < 0.0f;
            word_result |= value << in_word_i;

            has_voxel = has_voxel || (value != 0);
            has_air_nx = has_air_nx || (xi == 0 && value == 0);
            has_air_px = has_air_px || (xi == (VOXEL_BRICK_SIZE - 1) && value == 0);
            has_air_ny = has_air_ny || (yi == 0 && value == 0);
            has_air_py = has_air_py || (yi == (VOXEL_BRICK_SIZE - 1) && value == 0);
            has_air_nz = has_air_nz || (zi == 0 && value == 0);
            has_air_pz = has_air_pz || (zi == (VOXEL_BRICK_SIZE - 1) && value == 0);
        }
        bits[word_i] = word_result;
    }

    if (packmask(has_voxel) != 0) {
        *metadata |= (1 << 12);
    }
    if (packmask(has_air_nx) != 0) {
        *metadata |= (1 << 6);
    }
    if (packmask(has_air_px) != 0) {
        *metadata |= (1 << 9);
    }
    if (packmask(has_air_ny) != 0) {
        *metadata |= (1 << 7);
    }
    if (packmask(has_air_py) != 0) {
        *metadata |= (1 << 10);
    }
    if (packmask(has_air_nz) != 0) {
        *metadata |= (1 << 8);
    }
    if (packmask(has_air_pz) != 0) {
        *metadata |= (1 << 11);
    }
}

export void generate_attributes(
    uniform int brick_xi, uniform int brick_yi, uniform int brick_zi,
    uniform int chunk_xi, uniform int chunk_yi, uniform int chunk_zi,
    uniform int level_i, uniform uint packed_voxels[], // uniform float densities[],
    uniform NoiseSettings const *uniform noise_settings, RandomCtx random_ctx) {
    const uniform vec3 UP = {0, 0, 1};
    const uniform vec3 GRASS_COL = {12, 163, 7};
    const uniform vec3 DIRT_COL = {112, 62, 30};
    const uniform vec3 STONE_COL = {140, 110, 100};

    foreach (word_i = 0 ... (VOXEL_BRICK_SIZE * VOXEL_BRICK_SIZE * VOXEL_BRICK_SIZE / 32)) {
        for (uniform uint in_word_i = 0; in_word_i < 32; ++in_word_i) {
            uint voxel_index = word_i * 32 + in_word_i;
            int xi = voxel_index & ((1 << VOXEL_BRICK_SIZE_LOG2) - 1);
            int yi = (voxel_index >> VOXEL_BRICK_SIZE_LOG2) & ((1 << VOXEL_BRICK_SIZE_LOG2) - 1);
            int zi = (voxel_index >> (VOXEL_BRICK_SIZE_LOG2 * 2)) & ((1 << VOXEL_BRICK_SIZE_LOG2) - 1);

            vec3 pos;
            pos.x = (xi + brick_xi * VOXEL_BRICK_SIZE + chunk_xi * VOXEL_CHUNK_SIZE) * (1 << level_i);
            pos.y = (yi + brick_yi * VOXEL_BRICK_SIZE + chunk_yi * VOXEL_CHUNK_SIZE) * (1 << level_i);
            pos.z = (zi + brick_zi * VOXEL_BRICK_SIZE + chunk_zi * VOXEL_CHUNK_SIZE) * (1 << level_i);

            pos.x = (pos.x + 0.5f) * VOXEL_SIZE;
            pos.y = (pos.y + 0.5f) * VOXEL_SIZE;
            pos.z = (pos.z + 0.5f) * VOXEL_SIZE;

            DensityNrm dn = voxel_value(random_ctx, noise_settings, pos);
            Voxel voxel;
            if (dot(dn.nrm, UP) > 0.25f && dn.val > -2.5f) {
                voxel.col = GRASS_COL / 255.0f;
            } else if (dot(dn.nrm, UP) > 0.0f && dn.val > -7.5f) {
                voxel.col = DIRT_COL / 255.0f;
            } else {
                voxel.col = STONE_COL / 255.0f;
            }
            ivec3 o = {xi, yi, zi};
            voxel.nrm = dither_nrm(random_ctx, dn.nrm, o);
            PackedVoxel packed_voxel = pack_voxel(voxel);
            packed_voxels[voxel_index] = packed_voxel.data;
            // densities[voxel_index] = dn.val;
        }
    }
}

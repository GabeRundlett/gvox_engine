#include "common.hpp"

MinMax voxel_minmax_value_cpp(NoiseSettings const *noise_settings, RandomCtx random_ctx, float p0x, float p0y, float p0z, float p1x, float p1y, float p1z) {
    return voxel_minmax_value(random_ctx, noise_settings, vec3(p0x, p0y, p0z), vec3(p1x, p1y, p1z));
}

bool voxel_is_solid_cpp(NoiseSettings const *noise_settings, RandomCtx random_ctx, float px, float py, float pz) {
    return voxel_value(random_ctx, noise_settings, vec3(px, py, pz)).val < 0.0f;
}

void voxel_normal_cpp(NoiseSettings const *noise_settings, RandomCtx random_ctx, float px, float py, float pz, float out_normal[3]) {
    auto nrm = voxel_value(random_ctx, noise_settings, vec3(px, py, pz)).nrm;
    out_normal[0] = nrm.x;
    out_normal[1] = nrm.y;
    out_normal[2] = nrm.z;
}

void generate_bitmask_cpp(
    int brick_xi, int brick_yi, int brick_zi,
    int chunk_xi, int chunk_yi, int chunk_zi,
    int level_i, uint bits[], uint *uniform metadata,
    NoiseSettings const *noise_settings, RandomCtx random_ctx) {

    for (int zi = 0; zi < BRICK_SIZE; ++zi) {
        for (int yi = 0; yi < BRICK_SIZE; ++yi) {
            for (int xi = 0; xi < BRICK_SIZE; ++xi) {
                float x = (float((xi + brick_xi * BRICK_SIZE + chunk_xi * CHUNK_SIZE_VOXELS) * (1 << level_i)) + 0.5f) * VOXEL_SIZE;
                float y = (float((yi + brick_yi * BRICK_SIZE + chunk_yi * CHUNK_SIZE_VOXELS) * (1 << level_i)) + 0.5f) * VOXEL_SIZE;
                float z = (float((zi + brick_zi * BRICK_SIZE + chunk_zi * CHUNK_SIZE_VOXELS) * (1 << level_i)) + 0.5f) * VOXEL_SIZE;

                uint value = voxel_value(random_ctx, noise_settings, vec3(x, y, z)).val < 0.0f ? 1 : 0;

                if (value != 0) {
                    *metadata |= (1 << 22);
                }
                if (xi == 0 && value == 0) {
                    *metadata |= (1 << 16);
                } else if (xi == (BRICK_SIZE - 1) && value == 0) {
                    *metadata |= (1 << 19);
                }
                if (yi == 0 && value == 0) {
                    *metadata |= (1 << 17);
                } else if (yi == (BRICK_SIZE - 1) && value == 0) {
                    *metadata |= (1 << 20);
                }
                if (zi == 0 && value == 0) {
                    *metadata |= (1 << 18);
                } else if (zi == (BRICK_SIZE - 1) && value == 0) {
                    *metadata |= (1 << 21);
                }

                uint voxel_index = xi + yi * BRICK_SIZE + zi * BRICK_SIZE * BRICK_SIZE;
                uint voxel_word_index = voxel_index / 32;
                uint voxel_in_word_index = voxel_index % 32;
                bits[voxel_word_index] |= uint32_t(value) << voxel_in_word_index;
            }
        }
    }
}

void generate_attributes_cpp(
    int brick_xi, int brick_yi, int brick_zi,
    int chunk_xi, int chunk_yi, int chunk_zi,
    int level_i, uint packed_voxels[], uint foliage_bits[],
    NoiseSettings const *noise_settings, RandomCtx random_ctx) {

    const uniform vec3 GRASS_COL = {0.03, 0.08, 0.004};
    const uniform vec3 DIRT_COL = {0.34 * 0.34, 0.30 * 0.30, 0.14 * 0.14};
    const uniform vec3 STONE_COL = {0.33 * 0.33, 0.30 * 0.30, 0.21 * 0.21};
    const uniform vec3 GRAVEL_COL = {0.24 * 0.24, 0.18 * 0.18, 0.10 * 0.10};

    for (int zi = 0; zi < BRICK_SIZE; ++zi) {
        for (int yi = 0; yi < BRICK_SIZE; ++yi) {
            for (int xi = 0; xi < BRICK_SIZE; ++xi) {
                uint32_t voxel_index = xi + yi * BRICK_SIZE + zi * BRICK_SIZE * BRICK_SIZE;
                float x = (float((xi + brick_xi * BRICK_SIZE + chunk_xi * CHUNK_SIZE_VOXELS) * (1 << level_i)) + 0.5f) * VOXEL_SIZE;
                float y = (float((yi + brick_yi * BRICK_SIZE + chunk_yi * CHUNK_SIZE_VOXELS) * (1 << level_i)) + 0.5f) * VOXEL_SIZE;
                float z = (float((zi + brick_zi * BRICK_SIZE + chunk_zi * CHUNK_SIZE_VOXELS) * (1 << level_i)) + 0.5f) * VOXEL_SIZE;
                auto dn = voxel_value(random_ctx, noise_settings, glm::vec3(x, y, z));
                auto dn_above = voxel_value(random_ctx, noise_settings, glm::vec3(x, y, z + VOXEL_SIZE));
                uint self_is_solid = dn.val < 0.0f;
                uint above_is_solid = dn_above.val < 0.0f;
                Voxel voxel;
                float upwards = dot(dn.nrm, UP);
                float r = random_ctx[voxel_index] / 255.0f;
                if (dot(dn.nrm, UP) > 0.65f && dn.val > -2.5f) {
                    voxel.albedo = GRASS_COL;
                    voxel.roughness = 0.9;
                } else if (dn.val > -0.15 && upwards > 0.40) {
                    voxel.albedo = GRAVEL_COL;
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
                } else if (dn.val < -0.01 && dn.val > -0.07 && upwards > 0.2) {
                    voxel.albedo = DIRT_COL;
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
                packed_voxels[voxel_index] = pack_voxel(voxel).data;

                uint32_t word_i = voxel_index / 32;
                uint32_t in_word_i = voxel_index % 32;
                if (above_is_solid == 0 && self_is_solid != 0 && dot(dn.nrm, vec3(0, 0, 1)) > 0.85f && r > 0.25f)
                    foliage_bits[word_i] |= 1 << in_word_i;
                else
                    foliage_bits[word_i] &= ~(1 << in_word_i);
                // densities[voxel_index] = dn.val;
            }
        }
    }
}

float generate_upwards(
    int brick_xi, int brick_yi, int brick_zi,
    int chunk_xi, int chunk_yi, int chunk_zi,
    int level_i,
    NoiseSettings const *noise_settings, RandomCtx random_ctx) {

    float x = (float(((0.5f + brick_xi) * BRICK_SIZE + chunk_xi * CHUNK_SIZE_VOXELS) * (1 << level_i)) + 0.5f) * VOXEL_SIZE;
    float y = (float(((0.5f + brick_yi) * BRICK_SIZE + chunk_yi * CHUNK_SIZE_VOXELS) * (1 << level_i)) + 0.5f) * VOXEL_SIZE;
    float z = (float(((0.5f + brick_zi) * BRICK_SIZE + chunk_zi * CHUNK_SIZE_VOXELS) * (1 << level_i)) + 0.5f) * VOXEL_SIZE;

    DensityNrm val = voxel_value(random_ctx, noise_settings, glm::vec3(x, y, z));
    return dot(val.nrm, glm::vec3(0, 0, 1));
}

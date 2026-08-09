#include "common.hpp"

MinMax voxel_minmax_value_cpp(NoiseSettings const *noise_settings, RandomCtx random_ctx, float p0x, float p0y, float p0z, float p1x, float p1y, float p1z) {
    return voxel_minmax_value(random_ctx, noise_settings, vec3(p0x, p0y, p0z), vec3(p1x, p1y, p1z));
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
                    *metadata |= (1 << 12);
                }
                if (xi == 0 && value == 0) {
                    *metadata |= (1 << 6);
                } else if (xi == (BRICK_SIZE - 1) && value == 0) {
                    *metadata |= (1 << 9);
                }
                if (yi == 0 && value == 0) {
                    *metadata |= (1 << 7);
                } else if (yi == (BRICK_SIZE - 1) && value == 0) {
                    *metadata |= (1 << 10);
                }
                if (zi == 0 && value == 0) {
                    *metadata |= (1 << 8);
                } else if (zi == (BRICK_SIZE - 1) && value == 0) {
                    *metadata |= (1 << 11);
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
    int level_i, uint packed_voxels[], // float densities[],
    NoiseSettings const *noise_settings, RandomCtx random_ctx) {

    for (int zi = 0; zi < BRICK_SIZE; ++zi) {
        for (int yi = 0; yi < BRICK_SIZE; ++yi) {
            for (int xi = 0; xi < BRICK_SIZE; ++xi) {
                uint32_t voxel_index = xi + yi * BRICK_SIZE + zi * BRICK_SIZE * BRICK_SIZE;
                float x = (float((xi + brick_xi * BRICK_SIZE + chunk_xi * CHUNK_SIZE_VOXELS) * (1 << level_i)) + 0.5f) * VOXEL_SIZE;
                float y = (float((yi + brick_yi * BRICK_SIZE + chunk_yi * CHUNK_SIZE_VOXELS) * (1 << level_i)) + 0.5f) * VOXEL_SIZE;
                float z = (float((zi + brick_zi * BRICK_SIZE + chunk_zi * CHUNK_SIZE_VOXELS) * (1 << level_i)) + 0.5f) * VOXEL_SIZE;
                auto dn = voxel_value(random_ctx, noise_settings, glm::vec3(x, y, z));
                auto col = glm::vec3(0.0f);
                if (dot(dn.nrm, vec3(0, 0, 1)) > 0.5f && dn.val > -0.5f) {
                    col = glm::vec3(12, 163, 7) / 255.0f;
                } else {
                    col = glm::vec3(112, 62, 30) / 255.0f;
                }
                ivec3 o = {xi, yi, zi};
                dn.nrm = dither_nrm(random_ctx, dn.nrm, o);
                packed_voxels[voxel_index] = pack_voxel(Voxel(col, dn.nrm, 0.5, 0)).data;
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

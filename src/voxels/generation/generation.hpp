#pragma once

#include <voxels/generation/defs.inl>
#include <base/profiler.hpp>

MinMax voxel_minmax_value_cpp(NoiseSettings const *noise_settings, RandomCtx random_ctx, float p0x, float p0y, float p0z, float p1x, float p1y, float p1z);
bool voxel_is_solid_cpp(NoiseSettings const *noise_settings, RandomCtx random_ctx, float px, float py, float pz);
void voxel_normal_cpp(NoiseSettings const *noise_settings, RandomCtx random_ctx, float px, float py, float pz, float out_normal[3]);

#if USE_ISPC
#include <generation_ispc.h>
#else

void generate_bitmask_cpp(
    int brick_xi, int brick_yi, int brick_zi,
    int chunk_xi, int chunk_yi, int chunk_zi,
    int level_i, unsigned int bits[], unsigned int *metadata,
    NoiseSettings const *noise_settings, RandomCtx random_ctx);

void generate_attributes_cpp(
    int brick_xi, int brick_yi, int brick_zi,
    int chunk_xi, int chunk_yi, int chunk_zi,
    int level_i, unsigned int packed_voxels[], // float densities[],
    NoiseSettings const *noise_settings, RandomCtx random_ctx);

#endif

static inline void generate_bitmask(
    int brick_xi, int brick_yi, int brick_zi,
    int chunk_xi, int chunk_yi, int chunk_zi,
    int level_i, unsigned int bits[], unsigned int *metadata,
    NoiseSettings const *noise_settings, RandomCtx random_ctx) {
    PROFILE_FUNC();

#if USE_ISPC
    ispc::generate_bitmask(brick_xi, brick_yi, brick_zi, chunk_xi, chunk_yi, chunk_zi, level_i, bits, metadata, reinterpret_cast<ispc::NoiseSettings const *>(noise_settings), random_ctx);
#else
    generate_bitmask_cpp(brick_xi, brick_yi, brick_zi, chunk_xi, chunk_yi, chunk_zi, level_i, bits, metadata, noise_settings, random_ctx);
#endif
}

static inline void generate_attributes(
    int brick_xi, int brick_yi, int brick_zi,
    int chunk_xi, int chunk_yi, int chunk_zi,
    int level_i, unsigned int packed_voxels[], unsigned int foliage_bits[],
    NoiseSettings const *noise_settings, RandomCtx random_ctx) {
    PROFILE_FUNC();

#if USE_ISPC
    ispc::generate_attributes(brick_xi, brick_yi, brick_zi, chunk_xi, chunk_yi, chunk_zi, level_i, packed_voxels, foliage_bits, reinterpret_cast<ispc::NoiseSettings const *>(noise_settings), random_ctx);
#else
    generate_attributes_cpp(brick_xi, brick_yi, brick_zi, chunk_xi, chunk_yi, chunk_zi, level_i, packed_voxels, foliage_bits, noise_settings, random_ctx);
#endif
}

float generate_upwards(
    int brick_xi, int brick_yi, int brick_zi,
    int chunk_xi, int chunk_yi, int chunk_zi,
    int level_i,
    NoiseSettings const *noise_settings, RandomCtx random_ctx);

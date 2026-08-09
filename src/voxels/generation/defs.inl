#pragma once

#include "../defs.inl"

#define CHUNK_SIZE_VOXELS 64
#define CHUNK_SIZE_BRICKS (CHUNK_SIZE_VOXELS / BRICK_SIZE)

#define RANDOM_BUFFER_SIZE_LOG2 8
#define RANDOM_BUFFER_SIZE (1 << RANDOM_BUFFER_SIZE_LOG2)

#if defined(__cplusplus)
#undef VOXEL_SCL
#define VOXEL_SCL float(1 << LOG2_VOXELS_PER_METER)
using RandomCtx = unsigned char const *;
#elif ISPC
#define RandomCtx const uint8 *uniform
#endif

struct MinMax {
    float min;
    float max;
};

struct NoiseSettings {
    float persistence;
    float lacunarity;
    float scale;
    float amplitude;
    int octaves;
};

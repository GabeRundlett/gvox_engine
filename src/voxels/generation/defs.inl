#ifndef VOXELS_GENERATION_DEFS_INL
#define VOXELS_GENERATION_DEFS_INL

#include "../defs.inl"

#define CHUNK_SIZE_VOXELS_LOG2 7
#define CHUNK_SIZE_VOXELS (1 << CHUNK_SIZE_VOXELS_LOG2)
#define CHUNK_SIZE_BRICKS (CHUNK_SIZE_VOXELS / BRICK_SIZE)
#define CHUNK_SIZE_BRICKS_LOG2 (CHUNK_SIZE_VOXELS_LOG2 - BRICK_SIZE_LOG2)

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

#endif // VOXELS_GENERATION_DEFS_INL

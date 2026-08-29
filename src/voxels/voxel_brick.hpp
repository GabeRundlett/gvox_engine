#pragma once

#include <glm/vec3.hpp>
#include <base/vec.hpp>
#include <voxels/defs.inl>

struct VoxelBrick {
    glm::u8vec3 voxel_min, voxel_max;
    glm::ivec3 brick_i;
    uint64_t bitmask[BRICK_SIZE * BRICK_SIZE * BRICK_SIZE / 64];
    uint64_t foliage_bitmask[BRICK_SIZE * BRICK_SIZE * BRICK_SIZE / 64];
    struct VoxelShadingAttribBrick *render_attribs;
    uint32_t metadata;
};

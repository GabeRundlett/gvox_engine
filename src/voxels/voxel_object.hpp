#pragma once

#include <glm/vec3.hpp>
#include <vector>

constexpr auto BRICK_SIZE = 8u;

struct VoxelBrick {
    glm::u8vec3 voxel_min, voxel_max;
    glm::ivec3 brick_i;
    uint64_t bitmask[BRICK_SIZE * BRICK_SIZE * BRICK_SIZE / 64];
    
    
};

struct VoxelObject {
    glm::ivec3 voxel_min, voxel_max;
    glm::ivec3 brick_min, brick_max;
    float scale = 1.0f;
    std::vector<VoxelBrick *> bricks;

    ~VoxelObject();

    VoxelObject(const VoxelObject &) = delete;
    VoxelObject(VoxelObject &&) = delete;
    auto operator=(const VoxelObject &) -> VoxelObject& = delete;
    auto operator=(VoxelObject &&) -> VoxelObject& = delete;
};

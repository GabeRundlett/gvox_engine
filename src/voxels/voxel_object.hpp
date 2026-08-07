#pragma once

#include <glm/vec3.hpp>
#include <vector>
#include <voxels/defs.inl>

struct VoxelRenderBrick {
    uint64_t packed_attribs[BRICK_SIZE * BRICK_SIZE * BRICK_SIZE];
};

struct VoxelBrick {
    glm::u8vec3 voxel_min, voxel_max;
    glm::ivec3 brick_i;
    uint64_t bitmask[BRICK_SIZE * BRICK_SIZE * BRICK_SIZE / 64];
    VoxelRenderBrick *render_attribs;
};

struct VoxelObject {
    glm::ivec3 voxel_min, voxel_max;
    glm::ivec3 brick_min, brick_max;
    float scale = 1.0f;
    std::vector<VoxelBrick *> brick_grid;
    struct RenderVoxelObject *render_voxel_object;
    bool render_dirty = true;

    VoxelObject() = default;
    ~VoxelObject();

    VoxelObject(const VoxelObject &) = delete;
    VoxelObject(VoxelObject &&) = delete;
    auto operator=(const VoxelObject &) -> VoxelObject & = delete;
    auto operator=(VoxelObject &&) -> VoxelObject & = delete;
};

#pragma once

#include <glm/vec3.hpp>
#include <vector>
#include <voxels/defs.inl>

struct VoxelBrick {
    glm::u8vec3 voxel_min, voxel_max;
    glm::ivec3 brick_i;
    uint64_t bitmask[BRICK_SIZE * BRICK_SIZE * BRICK_SIZE / 64];
    struct VoxelShadingAttribBrick *render_attribs;
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

    void init(const glm::ivec3 &new_voxel_min, const glm::ivec3 &new_voxel_max);
    void resize(const glm::ivec3 &new_voxel_min, const glm::ivec3 &new_voxel_max);

    static inline glm::ivec3 get_brick_coord(const glm::ivec3 &voxel_coord) { return voxel_coord >> BRICK_SIZE_LOG2; }
    static inline glm::ivec3 get_voxel_offset(const glm::ivec3 &voxel_coord) { return voxel_coord & BRICK_MASK; }
    static inline int get_brick_index(const glm::ivec3 &brick_coord, const glm::ivec3 &brick_min, const glm::ivec3 &brick_max) {
        auto brick_i = brick_coord - brick_min;
        auto grid_size = brick_max - brick_min + 1;
        return brick_i.x + brick_i.y * grid_size.x + brick_i.z * grid_size.x * grid_size.y;
    }
    int get_brick_index(const glm::ivec3 &brick_coord) const { return get_brick_index(brick_coord, brick_min, brick_max); }
};

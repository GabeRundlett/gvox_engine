#pragma once

struct VoxelObject {
    glm::ivec3 voxel_min, voxel_max;
    glm::ivec3 brick_min, brick_max;
    Vec<struct VoxelBrick *> brick_grid;
    struct RenderVoxelObject *render_voxel_object;
    bool render_dirty = true;
    struct VoxelAllocator *allocator = nullptr;

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

    struct VoxelBrick *alloc_brick();
    void free_brick(struct VoxelBrick *brick);
    struct VoxelShadingAttribBrick *alloc_render_brick();
    void free_render_brick(struct VoxelShadingAttribBrick *render_brick);
};

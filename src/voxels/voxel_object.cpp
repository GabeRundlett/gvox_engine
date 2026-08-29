#include "voxel_object.hpp"
#include <glm/common.hpp>
#include <glm/vector_relational.hpp>
#include <base/profiler.hpp>
#include "voxel_brick.hpp"
#include "voxel_allocator.hpp"

VoxelObject::~VoxelObject() {
    PROFILE_FUNC();
    for (auto *brick : brick_grid) {
        if (brick != nullptr)
            free_brick(brick);
    }
}

void VoxelObject::init(const glm::ivec3 &new_voxel_min, const glm::ivec3 &new_voxel_max) {
    brick_min = get_brick_coord(new_voxel_min);
    brick_max = get_brick_coord(new_voxel_max);

    if (glm::any(glm::lessThanEqual(brick_min, brick_max))) {
        brick_grid.resize((brick_max.x - brick_min.x + 1) * (brick_max.y - brick_min.y + 1) * (brick_max.z - brick_min.z + 1));
    } else {
        brick_grid.clear();
    }
}

void VoxelObject::resize(const glm::ivec3 &new_voxel_min, const glm::ivec3 &new_voxel_max) {
    PROFILE_FUNC();

    auto new_brick_min = get_brick_coord(new_voxel_min);
    auto new_brick_max = get_brick_coord(new_voxel_max);

    if (new_brick_min != brick_min || new_brick_max != brick_max) {
        auto old_brick_min = brick_min;
        auto old_brick_max = brick_max;
        auto old_brick_grid = std::move(brick_grid);
        init(new_voxel_min, new_voxel_max);
        auto copy_min = glm::max(old_brick_min, brick_min);
        auto copy_max = glm::min(old_brick_max, brick_max);
        for (int z = copy_min.z; z <= copy_max.z; z++) {
            for (int y = copy_min.y; y <= copy_max.y; y++) {
                for (int x = copy_min.x; x <= copy_max.x; x++) {
                    int old_index = get_brick_index(glm::ivec3(x, y, z), old_brick_min, old_brick_max);
                    int new_index = get_brick_index(glm::ivec3(x, y, z));
                    brick_grid[new_index] = old_brick_grid[old_index];
                }
            }
        }
    }

    voxel_min = new_voxel_min;
    voxel_max = new_voxel_max;
}

VoxelBrick *VoxelObject::alloc_brick() {
    if (allocator != nullptr)
        return ::alloc_brick(allocator);
    else
        return new VoxelBrick();
}

void VoxelObject::free_brick(VoxelBrick *brick) {
    if (allocator != nullptr)
        ::free_brick(allocator, brick);
    else
        delete brick;
}

VoxelShadingAttribBrick *VoxelObject::alloc_render_brick() {
    if (allocator != nullptr)
        return ::alloc_render_brick(allocator);
    else
        return new VoxelShadingAttribBrick();
}

void VoxelObject::free_render_brick(VoxelShadingAttribBrick *brick) {
    if (allocator != nullptr)
        ::free_render_brick(allocator, brick);
    else
        delete brick;
}

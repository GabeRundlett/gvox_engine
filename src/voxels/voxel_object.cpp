#include "voxel_object.hpp"

VoxelObject::~VoxelObject() {
    for (auto *brick : brick_grid)
        if (brick != nullptr)
            delete brick;
}

#pragma once

#include <shared/voxels.inl>

#define MAX_SPHERES 10
#define MAX_BOXES 10
#define MAX_CAPSULES 10

struct Scene {
    u32 sphere_n;
    u32 box_n;
    u32 capsule_n;

    Sphere spheres[MAX_SPHERES];
    Box boxes[MAX_BOXES];
    Capsule capsules[MAX_BOXES];
    VoxelWorld voxel_world;
};

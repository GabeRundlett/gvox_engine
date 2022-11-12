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
    BoundingBox boxes[MAX_BOXES];
    CapsulePoints capsules[MAX_CAPSULES];

    BoundingBox pick_box;
    Sphere brush_origin_sphere;
};

struct Ray {
    f32vec3 o;
    f32vec3 nrm;
    f32vec3 inv_nrm;
};

struct IntersectionRecord {
    b32 hit;
    f32 internal_fac;
    f32 dist;
    f32vec3 nrm;
};

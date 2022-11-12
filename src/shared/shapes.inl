#pragma once

#include <daxa/daxa.inl>

struct Sphere {
    f32vec3 o;
    f32 r;
};

struct BoundingBox {
    f32vec3 bound_min, bound_max;
};

struct CapsulePoints {
    f32vec3 p0, p1;
    f32 r;
};

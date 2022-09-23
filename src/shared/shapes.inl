#pragma once

#include <daxa/daxa.inl>

struct Sphere {
    f32vec3 o;
    f32 r;
};

struct Box {
    f32vec3 bound_min, bound_max;
};

struct Capsule {
    f32vec3 p0, p1;
    f32 r;
};

b32 inside(f32vec3 p, Sphere s) {
    return dot(p - s.o, p - s.o) < s.r * s.r;
}
b32 inside(f32vec3 p, Box b) {
    return f32(p.x >= b.bound_min.x) * f32(p.y >= b.bound_min.y) * f32(p.z >= b.bound_min.z) * f32(p.x <= b.bound_max.x) * f32(p.y <= b.bound_max.y) * f32(p.z <= b.bound_max.z) > 0;
}

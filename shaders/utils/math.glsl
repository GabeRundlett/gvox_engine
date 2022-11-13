#pragma once

#define PI 3.14159265
#define MAX_SD 10000.0

f32 deg2rad(f32 d) {
    return d * PI / 180.0;
}

f32 rad2deg(f32 r) {
    return r * 180.0 / PI;
}

f32vec3 rotate_x(f32vec3 v, f32 angle) {
    float sin_rot_x = sin(angle), cos_rot_x = cos(angle);
    f32mat3x3 rot_mat = f32mat3x3(
        1, 0, 0,
        0, cos_rot_x, sin_rot_x,
        0, -sin_rot_x, cos_rot_x);
    return rot_mat * v;
}

f32vec3 rotate_y(f32vec3 v, f32 angle) {
    float sin_rot_y = sin(angle), cos_rot_y = cos(angle);
    f32mat3x3 rot_mat = f32mat3x3(
        cos_rot_y, 0, sin_rot_y,
        0, 1, 0,
        -sin_rot_y, 0, cos_rot_y);
    return rot_mat * v;
}

f32vec3 rotate_z(f32vec3 v, f32 angle) {
    float sin_rot_z = sin(angle), cos_rot_z = cos(angle);
    f32mat3x3 rot_mat = f32mat3x3(
        cos_rot_z, -sin_rot_z, 0,
        sin_rot_z, cos_rot_z, 0,
        0, 0, 1);
    return rot_mat * v;
}

f32vec3 rgb2hsv(f32vec3 c) {
    f32vec4 K = f32vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    f32vec4 p = mix(f32vec4(c.bg, K.wz), f32vec4(c.gb, K.xy), step(c.b, c.g));
    f32vec4 q = mix(f32vec4(p.xyw, c.r), f32vec4(c.r, p.yzx), step(p.x, c.r));
    f32 d = q.x - min(q.w, q.y);
    f32 e = 1.0e-10;
    return f32vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

f32vec3 hsv2rgb(f32vec3 c) {
    f32vec4 k = f32vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    f32vec3 p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * mix(k.xxx, clamp(p - k.xxx, 0.0, 1.0), c.y);
}

f32vec4 uint_to_float4(u32 u) {
    f32vec4 result;
    result.r = f32((u >> 0x00) & 0xff) / 255;
    result.g = f32((u >> 0x08) & 0xff) / 255;
    result.b = f32((u >> 0x10) & 0xff) / 255;
    result.a = f32((u >> 0x18) & 0xff) / 255;
    return result;
}

u32 float4_to_uint(f32vec4 f) {
    u32 result = 0;
    result |= u32(clamp(f.r, 0, 1) * 255) << 0x00;
    result |= u32(clamp(f.g, 0, 1) * 255) << 0x08;
    result |= u32(clamp(f.b, 0, 1) * 255) << 0x10;
    result |= u32(clamp(f.a, 0, 1) * 255) << 0x18;
    return result;
}

b32 inside(f32vec3 p, Sphere s) {
    return dot(p - s.o, p - s.o) < s.r * s.r;
}

b32 inside(f32vec3 p, BoundingBox b) {
    return f32(p.x >= b.bound_min.x) * f32(p.y >= b.bound_min.y) * f32(p.z >= b.bound_min.z) * f32(p.x <= b.bound_max.x) * f32(p.y <= b.bound_max.y) * f32(p.z <= b.bound_max.z) > 0;
}

b32 overlaps(BoundingBox a, BoundingBox b) {
    b32 x_overlap = a.bound_max.x >= b.bound_min.x && b.bound_max.x >= a.bound_min.x;
    b32 y_overlap = a.bound_max.y >= b.bound_min.y && b.bound_max.y >= a.bound_min.y;
    b32 z_overlap = a.bound_max.z >= b.bound_min.z && b.bound_max.z >= a.bound_min.z;
    return x_overlap && y_overlap && z_overlap;
}

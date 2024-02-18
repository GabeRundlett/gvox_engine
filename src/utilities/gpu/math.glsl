#pragma once

#include <utilities/gpu/defs.glsl>
#include <renderer/kajiya/inc/math_const.glsl>

// Definitions

#define MAX_STEPS 512
const float MAX_DIST = 1.0e9;

// Objects

struct Sphere {
    vec3 o;
    float r;
};
struct BoundingBox {
    vec3 bound_min, bound_max;
};
struct CapsulePoints {
    vec3 p0, p1;
    float r;
};
struct Aabb {
    vec3 pmin;
    vec3 pmax;
};

#include <utilities/gpu/common.glsl>

vec3 rotate_x(vec3 v, float angle) {
    float sin_rot_x = sin(angle), cos_rot_x = cos(angle);
    mat3 rot_mat = mat3(
        1, 0, 0,
        0, cos_rot_x, sin_rot_x,
        0, -sin_rot_x, cos_rot_x);
    return rot_mat * v;
}
vec3 rotate_y(vec3 v, float angle) {
    float sin_rot_y = sin(angle), cos_rot_y = cos(angle);
    mat3 rot_mat = mat3(
        cos_rot_y, 0, sin_rot_y,
        0, 1, 0,
        -sin_rot_y, 0, cos_rot_y);
    return rot_mat * v;
}
vec3 rotate_z(vec3 v, float angle) {
    float sin_rot_z = sin(angle), cos_rot_z = cos(angle);
    mat3 rot_mat = mat3(
        cos_rot_z, -sin_rot_z, 0,
        sin_rot_z, cos_rot_z, 0,
        0, 0, 1);
    return rot_mat * v;
}

// Color functions
vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv2rgb(vec3 c) {
    vec4 k = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * mix(k.xxx, clamp(p - k.xxx, 0.0, 1.0), c.y);
}
vec4 uint_rgba8_to_f32vec4(uint u) {
    vec4 result;
    result.r = float((u >> 0x00) & 0xff) / 255.0;
    result.g = float((u >> 0x08) & 0xff) / 255.0;
    result.b = float((u >> 0x10) & 0xff) / 255.0;
    result.a = float((u >> 0x18) & 0xff) / 255.0;

    result = pow(result, vec4(2.2));
    return result;
}
uint vec4_to_uint_rgba8(vec4 f) {
    f = clamp(f, vec4(0), vec4(1));
    f = pow(f, vec4(1.0 / 2.2));

    uint result = 0;
    result |= uint(clamp(f.r, 0, 1) * 255) << 0x00;
    result |= uint(clamp(f.g, 0, 1) * 255) << 0x08;
    result |= uint(clamp(f.b, 0, 1) * 255) << 0x10;
    result |= uint(clamp(f.a, 0, 1) * 255) << 0x18;
    return result;
}

float uint_to_u01_float(uint h) {
    const uint mantissaMask = 0x007FFFFFu;
    const uint one = 0x3F800000u;

    h &= mantissaMask;
    h |= one;

    float r2 = uintBitsToFloat(h);
    return r2 - 1.0;
}

// [Drobot2014a] Low Level Optimizations for GCN
float fast_sqrt(float x) {
    return uintBitsToFloat(0x1fbd1df5 + (floatBitsToUint(x) >> 1u));
}

// [Eberly2014] GPGPU Programming for Games and Science
float fast_acos(float inX) {
    float x = abs(inX);
    float res = -0.156583f * x + M_PI * 0.5;
    res *= fast_sqrt(1.0f - x);
    return (inX >= 0) ? res : (M_PI - res);
}

float inverse_depth_relative_diff(float primary_depth, float secondary_depth) {
    return abs(max(1e-20, primary_depth) / max(1e-20, secondary_depth) - 1.0);
}

float exponential_squish(float len, float squish_scale) {
    return exp2(-clamp(squish_scale * len, 0, 100));
}

float exponential_unsquish(float len, float squish_scale) {
    return max(0.0, -1.0 / squish_scale * log2(1e-30 + len));
}

#define URGB9E5_CONCENTRATION 4.0
#define URGB9E5_MIN_EXPONENT -8.0
float urgb9e5_scale_exp_inv(float x) { return (exp((x + URGB9E5_MIN_EXPONENT) / URGB9E5_CONCENTRATION)); }
vec3 uint_urgb9e5_to_f32vec3(uint u) {
    vec3 result;
    result.r = float((u >> 0x00) & 0x1ff);
    result.g = float((u >> 0x09) & 0x1ff);
    result.b = float((u >> 0x12) & 0x1ff);
    float scale = urgb9e5_scale_exp_inv((u >> 0x1b) & 0x1f) / 511.0;
    return result * scale;
}
float urgb9e5_scale_exp(float x) { return URGB9E5_CONCENTRATION * log(x) - URGB9E5_MIN_EXPONENT; }
uint vec3_to_uint_urgb9e5(vec3 f) {
    float scale = max(max(f.x, 0.0), max(f.y, f.z));
    float exponent = ceil(clamp(urgb9e5_scale_exp(scale), 0, 31));
    float fac = 511.0 / urgb9e5_scale_exp_inv(exponent);
    uint result = 0;
    result |= uint(clamp(f.r * fac, 0.0, 511.0)) << 0x00;
    result |= uint(clamp(f.g * fac, 0.0, 511.0)) << 0x09;
    result |= uint(clamp(f.b * fac, 0.0, 511.0)) << 0x12;
    result |= uint(exponent) << 0x1b;
    return result;
}

#include <utilities/gpu/normal.glsl>

uint ceil_log2(uint x) {
    return findMSB(x) + uint(bitCount(x) > 1);
}

// Bit Functions

void flag_set(in out uint bitfield, uint index, bool value) {
    if (value) {
        bitfield |= 1u << index;
    } else {
        bitfield &= ~(1u << index);
    }
}
bool flag_get(uint bitfield, uint index) {
    return ((bitfield >> index) & 1u) == 1u;
}

// Shape Functions

bool inside(vec3 p, Sphere s) {
    return dot(p - s.o, p - s.o) < s.r * s.r;
}
bool inside(vec3 p, BoundingBox b) {
    return (p.x >= b.bound_min.x && p.x < b.bound_max.x &&
            p.y >= b.bound_min.y && p.y < b.bound_max.y &&
            p.z >= b.bound_min.z && p.z < b.bound_max.z);
}
bool overlaps(BoundingBox a, BoundingBox b) {
    bool x_overlap = a.bound_max.x >= b.bound_min.x && b.bound_max.x >= a.bound_min.x;
    bool y_overlap = a.bound_max.y >= b.bound_min.y && b.bound_max.y >= a.bound_min.y;
    bool z_overlap = a.bound_max.z >= b.bound_min.z && b.bound_max.z >= a.bound_min.z;
    return x_overlap && y_overlap && z_overlap;
}
void intersect(in out vec3 ray_pos, vec3 ray_dir, vec3 inv_dir, BoundingBox b) {
    if (inside(ray_pos, b)) {
        return;
    }
    float tx1 = (b.bound_min.x - ray_pos.x) * inv_dir.x;
    float tx2 = (b.bound_max.x - ray_pos.x) * inv_dir.x;
    float tmin = min(tx1, tx2);
    float tmax = max(tx1, tx2);
    float ty1 = (b.bound_min.y - ray_pos.y) * inv_dir.y;
    float ty2 = (b.bound_max.y - ray_pos.y) * inv_dir.y;
    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));
    float tz1 = (b.bound_min.z - ray_pos.z) * inv_dir.z;
    float tz2 = (b.bound_max.z - ray_pos.z) * inv_dir.z;
    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));

    // float dist = max(min(tmax, tmin), 0);
    float dist = MAX_DIST;
    if (tmax >= tmin) {
        if (tmin > 0) {
            dist = tmin;
        }
    }

    ray_pos = ray_pos + ray_dir * dist;
}

#include <utilities/gpu/signed_distance.glsl>
#include <utilities/gpu/random.glsl>

mat3 tbn_from_normal(vec3 nrm) {
    vec3 tangent = normalize(cross(nrm, -nrm.zxy));
    vec3 bi_tangent = cross(nrm, tangent);
    return mat3(tangent, bi_tangent, nrm);
}

// Building an Orthonormal Basis, Revisited
// http://jcgt.org/published/0006/01/01/
mat3 build_orthonormal_basis(vec3 n) {
    vec3 b1;
    vec3 b2;

    if (n.z < 0.0) {
        const float a = 1.0 / (1.0 - n.z);
        const float b = n.x * n.y * a;
        b1 = vec3(1.0 - n.x * n.x * a, -b, n.x);
        b2 = vec3(b, n.y * n.y * a - 1.0, -n.y);
    } else {
        const float a = 1.0 / (1.0 + n.z);
        const float b = -n.x * n.y * a;
        b1 = vec3(1.0 - n.x * n.x * a, b, -n.x);
        b2 = vec3(b, 1.0 - n.y * n.y * a, -n.y);
    }

    return mat3(b1, b2, n);
}

vec3 uniform_sample_cone(vec2 urand, float cos_theta_max) {
    float cos_theta = (1.0 - urand.x) + urand.x * cos_theta_max;
    float sin_theta = sqrt(clamp(1.0 - cos_theta * cos_theta, 0.0, 1.0));
    float phi = urand.y * (M_PI * 2.0);
    return vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

vec3 uniform_sample_hemisphere(vec2 urand) {
    float phi = urand.y * 2.0 * M_PI;
    float cos_theta = 1.0 - urand.x;
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    return vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

vec2 rsi(vec3 r0, vec3 rd, float sr) {
    // ray-sphere intersection that assumes
    // the sphere is centered at the origin.
    // No intersection when result.x > result.y
    float a = dot(rd, rd);
    float b = 2.0 * dot(rd, r0);
    float c = dot(r0, r0) - (sr * sr);
    float d = (b * b) - 4.0 * a * c;
    if (d < 0.0)
        return vec2(1e5, -1e5);
    return vec2(
        (-b - sqrt(d)) / (2.0 * a),
        (-b + sqrt(d)) / (2.0 * a));
}

bool rectangles_overlap(vec3 a_min, vec3 a_max, vec3 b_min, vec3 b_max) {
    bool x_disjoint = (a_max.x < b_min.x) || (b_max.x < a_min.x);
    bool y_disjoint = (a_max.y < b_min.y) || (b_max.y < a_min.y);
    bool z_disjoint = (a_max.z < b_min.z) || (b_max.z < a_min.z);
    return !x_disjoint && !y_disjoint && !z_disjoint;
}

// https://www.shadertoy.com/view/cdSBRG
int imod(int x, int m) {
    return x >= 0 ? x % m : m - 1 - (-x - 1) % m;
}
ivec3 imod3(ivec3 p, int m) {
    return ivec3(imod(p.x, m), imod(p.y, m), imod(p.z, m));
}
ivec3 imod3(ivec3 p, ivec3 m) {
    return ivec3(imod(p.x, m.x), imod(p.y, m.y), imod(p.z, m.z));
}

mat3 CUBE_MAP_FACE_ROTATION(uint face) {
    switch (face) {
    case 0: return mat3(+0, +0, -1, +0, -1, +0, -1, +0, +0);
    case 1: return mat3(+0, +0, +1, +0, -1, +0, +1, +0, +0);
    case 2: return mat3(+1, +0, +0, +0, +0, +1, +0, -1, +0);
    case 3: return mat3(+1, +0, +0, +0, +0, -1, +0, +1, +0);
    case 4: return mat3(+1, +0, +0, +0, -1, +0, +0, +0, -1);
    default: return mat3(-1, +0, +0, +0, -1, +0, +0, +0, +1);
    }
}

#pragma once

#include <utils/defs.glsl>

// Definitions

#define PI 3.14159265
#define MAX_STEPS 512

const float MAX_DIST = 1.0e9;
const float SQRT_2 = 1.41421356237;

// Objects

struct Sphere {
    daxa_f32vec3 o;
    daxa_f32 r;
};
struct BoundingBox {
    daxa_f32vec3 bound_min, bound_max;
};
struct CapsulePoints {
    daxa_f32vec3 p0, p1;
    daxa_f32 r;
};

// Common Functions

daxa_f32 saturate(daxa_f32 x) { return clamp(x, 0.0, 1.0); }
daxa_f32vec2 saturate(daxa_f32vec2 x) { return clamp(x, daxa_f32vec2(0.0), daxa_f32vec2(1.0)); }
daxa_f32vec3 saturate(daxa_f32vec3 x) { return clamp(x, daxa_f32vec3(0.0), daxa_f32vec3(1.0)); }
daxa_f32vec4 saturate(daxa_f32vec4 x) { return clamp(x, daxa_f32vec4(0.0), daxa_f32vec4(1.0)); }
daxa_f32 nonzero_sign(daxa_f32 x) {
    if (x < 0.0)
        return -1.0;
    return 1.0;
}
daxa_f32vec2 nonzero_sign(daxa_f32vec2 x) {
    return daxa_f32vec2(nonzero_sign(x.x), nonzero_sign(x.y));
}
daxa_f32vec3 nonzero_sign(daxa_f32vec3 x) {
    return daxa_f32vec3(nonzero_sign(x.x), nonzero_sign(x.y), nonzero_sign(x.z));
}
daxa_f32vec4 nonzero_sign(daxa_f32vec4 x) {
    return daxa_f32vec4(nonzero_sign(x.x), nonzero_sign(x.y), nonzero_sign(x.z), nonzero_sign(x.w));
}
daxa_f32 deg2rad(daxa_f32 d) {
    return d * PI / 180.0;
}
daxa_f32 rad2deg(daxa_f32 r) {
    return r * 180.0 / PI;
}
daxa_f32vec3 rotate_x(daxa_f32vec3 v, daxa_f32 angle) {
    float sin_rot_x = sin(angle), cos_rot_x = cos(angle);
    daxa_f32mat3x3 rot_mat = daxa_f32mat3x3(
        1, 0, 0,
        0, cos_rot_x, sin_rot_x,
        0, -sin_rot_x, cos_rot_x);
    return rot_mat * v;
}
daxa_f32vec3 rotate_y(daxa_f32vec3 v, daxa_f32 angle) {
    float sin_rot_y = sin(angle), cos_rot_y = cos(angle);
    daxa_f32mat3x3 rot_mat = daxa_f32mat3x3(
        cos_rot_y, 0, sin_rot_y,
        0, 1, 0,
        -sin_rot_y, 0, cos_rot_y);
    return rot_mat * v;
}
daxa_f32vec3 rotate_z(daxa_f32vec3 v, daxa_f32 angle) {
    float sin_rot_z = sin(angle), cos_rot_z = cos(angle);
    daxa_f32mat3x3 rot_mat = daxa_f32mat3x3(
        cos_rot_z, -sin_rot_z, 0,
        sin_rot_z, cos_rot_z, 0,
        0, 0, 1);
    return rot_mat * v;
}

// Color functions
daxa_f32vec3 rgb2hsv(daxa_f32vec3 c) {
    daxa_f32vec4 K = daxa_f32vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    daxa_f32vec4 p = mix(daxa_f32vec4(c.bg, K.wz), daxa_f32vec4(c.gb, K.xy), step(c.b, c.g));
    daxa_f32vec4 q = mix(daxa_f32vec4(p.xyw, c.r), daxa_f32vec4(c.r, p.yzx), step(p.x, c.r));
    daxa_f32 d = q.x - min(q.w, q.y);
    daxa_f32 e = 1.0e-10;
    return daxa_f32vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
daxa_f32vec3 hsv2rgb(daxa_f32vec3 c) {
    daxa_f32vec4 k = daxa_f32vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    daxa_f32vec3 p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * mix(k.xxx, clamp(p - k.xxx, 0.0, 1.0), c.y);
}
daxa_f32vec4 uint_rgba8_to_f32vec4(daxa_u32 u) {
    daxa_f32vec4 result;
    result.r = daxa_f32((u >> 0x00) & 0xff) / 255.0;
    result.g = daxa_f32((u >> 0x08) & 0xff) / 255.0;
    result.b = daxa_f32((u >> 0x10) & 0xff) / 255.0;
    result.a = daxa_f32((u >> 0x18) & 0xff) / 255.0;

    result = pow(result, daxa_f32vec4(2.2));
    return result;
}
daxa_u32 daxa_f32vec4_to_uint_rgba8(daxa_f32vec4 f) {
    f = clamp(f, daxa_f32vec4(0), daxa_f32vec4(1));
    f = pow(f, daxa_f32vec4(1.0 / 2.2));

    daxa_u32 result = 0;
    result |= daxa_u32(clamp(f.r, 0, 1) * 255) << 0x00;
    result |= daxa_u32(clamp(f.g, 0, 1) * 255) << 0x08;
    result |= daxa_u32(clamp(f.b, 0, 1) * 255) << 0x10;
    result |= daxa_u32(clamp(f.a, 0, 1) * 255) << 0x18;
    return result;
}
daxa_f32vec3 sRGB_to_YCbCr(daxa_f32vec3 col) {
    return daxa_f32mat3x3(0.2126, 0.7152, 0.0722, -0.1146, -0.3854, 0.5, 0.5, -0.4542, -0.0458) * col;
}
daxa_f32vec3 YCbCr_to_sRGB(daxa_f32vec3 col) {
    return max(daxa_f32vec3(0.0), daxa_f32mat3x3(1.0, 0.0, 1.5748, 1.0, -0.1873, -.4681, 1.0, 1.8556, 0.0) * col);
}
float sRGB_to_luminance(daxa_f32vec3 col) {
    return dot(col, daxa_f32vec3(0.2126, 0.7152, 0.0722));
}
daxa_f32vec4 linear_rgb_to_crunched_luma_chroma(daxa_f32vec4 v) {
    v.rgb = sRGB_to_YCbCr(v.rgb);
    float k = sqrt(v.x) / max(1e-8, v.x);
    return daxa_f32vec4(v.rgb * k, v.a);
}
daxa_f32vec4 crunched_luma_chroma_to_linear_rgb(daxa_f32vec4 v) {
    v.rgb *= v.x;
    v.rgb = YCbCr_to_sRGB(v.rgb);
    return v;
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
    float res = -0.156583f * x + PI * 0.5;
    res *= fast_sqrt(1.0f - x);
    return (inX >= 0) ? res : (PI - res);
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
daxa_f32 urgb9e5_scale_exp_inv(daxa_f32 x) { return (exp((x + URGB9E5_MIN_EXPONENT) / URGB9E5_CONCENTRATION)); }
daxa_f32vec3 uint_urgb9e5_to_f32vec3(daxa_u32 u) {
    daxa_f32vec3 result;
    result.r = daxa_f32((u >> 0x00) & 0x1ff);
    result.g = daxa_f32((u >> 0x09) & 0x1ff);
    result.b = daxa_f32((u >> 0x12) & 0x1ff);
    daxa_f32 scale = urgb9e5_scale_exp_inv((u >> 0x1b) & 0x1f) / 511.0;
    return result * scale;
}
daxa_f32 urgb9e5_scale_exp(daxa_f32 x) { return URGB9E5_CONCENTRATION * log(x) - URGB9E5_MIN_EXPONENT; }
daxa_u32 daxa_f32vec3_to_uint_urgb9e5(daxa_f32vec3 f) {
    daxa_f32 scale = max(max(f.x, 0.0), max(f.y, f.z));
    daxa_f32 exponent = ceil(clamp(urgb9e5_scale_exp(scale), 0, 31));
    daxa_f32 fac = 511.0 / urgb9e5_scale_exp_inv(exponent);
    daxa_u32 result = 0;
    result |= daxa_u32(clamp(f.r * fac, 0.0, 511.0)) << 0x00;
    result |= daxa_u32(clamp(f.g * fac, 0.0, 511.0)) << 0x09;
    result |= daxa_u32(clamp(f.b * fac, 0.0, 511.0)) << 0x12;
    result |= daxa_u32(exponent) << 0x1b;
    return result;
}

// ----------------------------------------
// The MIT License
// Copyright 2017 Inigo Quilez
vec2 msign(vec2 v) {
    return vec2((v.x >= 0.0) ? 1.0 : -1.0,
                (v.y >= 0.0) ? 1.0 : -1.0);
}
uint packSnorm2x8(vec2 v) {
    uvec2 d = uvec2(round(127.5 + v * 127.5));
    return d.x | (d.y << 8u);
}
vec2 unpackSnorm2x8(uint d) {
    return vec2(uvec2(d, d >> 8) & 255) / 127.5 - 1.0;
}
uint octahedral_8(in vec3 nor) {
    nor.xy /= (abs(nor.x) + abs(nor.y) + abs(nor.z));
    nor.xy = (nor.z >= 0.0) ? nor.xy : (1.0 - abs(nor.yx)) * msign(nor.xy);
    uvec2 d = uvec2(round(7.5 + nor.xy * 7.5));
    return d.x | (d.y << 4u);
}
vec3 i_octahedral_8(uint data) {
    uvec2 iv = uvec2(data, data >> 4u) & 15u;
    vec2 v = vec2(iv) / 7.5 - 1.0;
    vec3 nor = vec3(v, 1.0 - abs(v.x) - abs(v.y)); // Rune Stubbe's version,
    float t = max(-nor.z, 0.0);                    // much faster than original
    nor.x += (nor.x > 0.0) ? -t : t;               // implementation of this
    nor.y += (nor.y > 0.0) ? -t : t;               // technique
    return normalize(nor);
}
uint octahedral_12(in vec3 nor) {
    nor.xy /= (abs(nor.x) + abs(nor.y) + abs(nor.z));
    nor.xy = (nor.z >= 0.0) ? nor.xy : (1.0 - abs(nor.yx)) * msign(nor.xy);
    uvec2 d = uvec2(round(31.5 + nor.xy * 31.5));
    return d.x | (d.y << 6u);
}
vec3 i_octahedral_12(uint data) {
    uvec2 iv = uvec2(data, data >> 6u) & 63u;
    vec2 v = vec2(iv) / 31.5 - 1.0;
    vec3 nor = vec3(v, 1.0 - abs(v.x) - abs(v.y)); // Rune Stubbe's version,
    float t = max(-nor.z, 0.0);                    // much faster than original
    nor.x += (nor.x > 0.0) ? -t : t;               // implementation of this
    nor.y += (nor.y > 0.0) ? -t : t;               // technique
    return normalize(nor);
}
uint octahedral_16(in vec3 nor) {
    nor /= (abs(nor.x) + abs(nor.y) + abs(nor.z));
    nor.xy = (nor.z >= 0.0) ? nor.xy : (1.0 - abs(nor.yx)) * msign(nor.xy);
    return packSnorm2x8(nor.xy);
}
vec3 i_octahedral_16(uint data) {
    vec2 v = unpackSnorm2x8(data);
    vec3 nor = vec3(v, 1.0 - abs(v.x) - abs(v.y));
    float t = max(-nor.z, 0.0);
    nor.x += (nor.x > 0.0) ? -t : t;
    nor.y += (nor.y > 0.0) ? -t : t;
    return nor;
}
uint spheremap_16(in vec3 nor) {
    vec2 v = nor.xy * inversesqrt(2.0 * nor.z + 2.0);
    return packSnorm2x8(v);
}
vec3 i_spheremap_16(uint data) {
    vec2 v = unpackSnorm2x8(data);
    float f = dot(v, v);
    return vec3(2.0 * v * sqrt(1.0 - f), 1.0 - 2.0 * f);
}
// ----------------------------------------

daxa_f32vec3 u16_to_nrm(daxa_u32 x) {
    return normalize(i_octahedral_16(x));
    // return i_spheremap_16(x);
}
daxa_f32vec3 u16_to_nrm_unnormalized(daxa_u32 x) {
    return i_octahedral_16(x);
    // return i_spheremap_16(x);
}
daxa_u32 nrm_to_u16(daxa_f32vec3 nrm) {
    return octahedral_16(nrm);
    // return spheremap_16(nrm);
}

float unpack_unorm(uint pckd, uint bitCount) {
    uint maxVal = (1u << bitCount) - 1;
    return float(pckd & maxVal) / maxVal;
}

uint pack_unorm(float val, uint bitCount) {
    uint maxVal = (1u << bitCount) - 1;
    return uint(clamp(val, 0.0, 1.0) * maxVal + 0.5);
}

float pack_normal_11_10_11(daxa_f32vec3 n) {
    uint pckd = 0;
    pckd += pack_unorm(n.x * 0.5 + 0.5, 11);
    pckd += pack_unorm(n.y * 0.5 + 0.5, 10) << 11;
    pckd += pack_unorm(n.z * 0.5 + 0.5, 11) << 21;
    return uintBitsToFloat(pckd);
}

daxa_f32vec3 unpack_normal_11_10_11(float pckd) {
    uint p = floatBitsToUint(pckd);
    return normalize(daxa_f32vec3(
                         unpack_unorm(p, 11),
                         unpack_unorm(p >> 11, 10),
                         unpack_unorm(p >> 21, 11)) *
                         2.0 -
                     1.0);
}

daxa_u32 ceil_log2(daxa_u32 x) {
    return findMSB(x) + daxa_u32(bitCount(x) > 1);
}

// Bit Functions

void flag_set(in out daxa_u32 bitfield, daxa_u32 index, bool value) {
    if (value) {
        bitfield |= 1u << index;
    } else {
        bitfield &= ~(1u << index);
    }
}
bool flag_get(daxa_u32 bitfield, daxa_u32 index) {
    return ((bitfield >> index) & 1u) == 1u;
}

// Shape Functions

daxa_b32 inside(daxa_f32vec3 p, Sphere s) {
    return dot(p - s.o, p - s.o) < s.r * s.r;
}
daxa_b32 inside(daxa_f32vec3 p, BoundingBox b) {
    return (p.x >= b.bound_min.x && p.x < b.bound_max.x &&
            p.y >= b.bound_min.y && p.y < b.bound_max.y &&
            p.z >= b.bound_min.z && p.z < b.bound_max.z);
}
daxa_b32 overlaps(BoundingBox a, BoundingBox b) {
    daxa_b32 x_overlap = a.bound_max.x >= b.bound_min.x && b.bound_max.x >= a.bound_min.x;
    daxa_b32 y_overlap = a.bound_max.y >= b.bound_min.y && b.bound_max.y >= a.bound_min.y;
    daxa_b32 z_overlap = a.bound_max.z >= b.bound_min.z && b.bound_max.z >= a.bound_min.z;
    return x_overlap && y_overlap && z_overlap;
}
void intersect(in out daxa_f32vec3 ray_pos, daxa_f32vec3 ray_dir, daxa_f32vec3 inv_dir, BoundingBox b) {
    if (inside(ray_pos, b)) {
        return;
    }
    daxa_f32 tx1 = (b.bound_min.x - ray_pos.x) * inv_dir.x;
    daxa_f32 tx2 = (b.bound_max.x - ray_pos.x) * inv_dir.x;
    daxa_f32 tmin = min(tx1, tx2);
    daxa_f32 tmax = max(tx1, tx2);
    daxa_f32 ty1 = (b.bound_min.y - ray_pos.y) * inv_dir.y;
    daxa_f32 ty2 = (b.bound_max.y - ray_pos.y) * inv_dir.y;
    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));
    daxa_f32 tz1 = (b.bound_min.z - ray_pos.z) * inv_dir.z;
    daxa_f32 tz2 = (b.bound_max.z - ray_pos.z) * inv_dir.z;
    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));

    // daxa_f32 dist = max(min(tmax, tmin), 0);
    daxa_f32 dist = MAX_DIST;
    if (tmax >= tmin) {
        if (tmin > 0) {
            dist = tmin;
        }
    }

    ray_pos = ray_pos + ray_dir * dist;
}

/// SIGNED DISTANCE FUNCTIONS

daxa_f32 sd_shapes_dot2(in daxa_f32vec2 v) { return dot(v, v); }
daxa_f32 sd_shapes_dot2(in daxa_f32vec3 v) { return dot(v, v); }
daxa_f32 sd_shapes_ndot(in daxa_f32vec2 a, in daxa_f32vec2 b) { return a.x * b.x - a.y * b.y; }

// Operators

// These are safe for min/max operations!
daxa_f32 sd_set(daxa_f32 a, daxa_f32 b) {
    return b;
}
daxa_f32vec4 sd_set(daxa_f32vec4 a, daxa_f32vec4 b) {
    return b;
}
daxa_f32 sd_add(daxa_f32 a, daxa_f32 b) {
    return (a + b);
}
daxa_f32 sd_union(in daxa_f32 a, in daxa_f32 b) {
    return min(a, b);
}

// These are either unsafe or unknown for min/max operations
daxa_f32 sd_smooth_union(in daxa_f32 a, in daxa_f32 b, in daxa_f32 k) {
    daxa_f32 h = clamp(0.5 + 0.5 * (a - b) / k, 0.0, 1.0);
    return mix(a, b, h) - k * h * (1.0 - h);
}
daxa_f32 sd_intersection(in daxa_f32 a, in daxa_f32 b) {
    return max(a, b);
}
daxa_f32 sd_smooth_intersection(in daxa_f32 a, in daxa_f32 b, in daxa_f32 k) {
    return sd_smooth_union(a, b, -k);
}
daxa_f32 sd_difference(in daxa_f32 a, in daxa_f32 b) {
    return sd_intersection(a, -b);
}
daxa_f32 sd_smooth_difference(in daxa_f32 a, in daxa_f32 b, in daxa_f32 k) {
    return sd_smooth_intersection(a, -b, k);
}
daxa_f32 sd_mul(daxa_f32 a, daxa_f32 b) {
    return (a * b);
}

// Shapes

// assumed sphere is at (0, 0, 0)
float sd_sphere_nearest(vec3 p, float r) {
    return length(p) - r;
}
float sd_sphere_furthest(vec3 p, float r) {
    return length(p) + r * 2.0;
}

// assumed box is centered at (0, 0, 0)
float sd_box_nearest(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}
float sd_box_furthest(vec3 p, vec3 b) {
    return length(b * nonzero_sign(p) + p);
}

// assumed sphere is at (0, 0, 0)
vec2 minmax_sd_sphere_in_region(vec3 region_center, vec3 region_size, float r) {
    float min_d = sd_box_nearest(-region_center, region_size);
    float max_d = sd_box_furthest(-region_center, region_size);
    return vec2(min_d, max_d) - r;
}

daxa_f32 sd_plane_x(in daxa_f32vec3 p) {
    return p.x;
}
daxa_f32 sd_plane_y(in daxa_f32vec3 p) {
    return p.y;
}
daxa_f32 sd_plane_z(in daxa_f32vec3 p) {
    return p.z;
}
daxa_f32 sd_sphere(in daxa_f32vec3 p, in daxa_f32 r) {
    return length(p) - r;
}
daxa_f32 sd_ellipsoid(in daxa_f32vec3 p, in daxa_f32vec3 r) {
    daxa_f32 k0 = length(p / r);
    daxa_f32 k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}
daxa_f32 sd_box(in daxa_f32vec3 p, in daxa_f32vec3 size) {
    daxa_f32vec3 d = abs(p) - size;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}
daxa_f32 sd_box(in daxa_f32vec3 p, in BoundingBox box) {
    return sd_box(p - (box.bound_max + box.bound_min) * 0.5, (box.bound_max - box.bound_min) * 0.5);
}
daxa_f32 sd_box_frame(in daxa_f32vec3 p, in daxa_f32vec3 b, in daxa_f32 e) {
    p = abs(p) - b;
    daxa_f32vec3 q = abs(p + e) - e;
    return min(
        min(length(max(daxa_f32vec3(p.x, q.y, q.z), 0.0)) + min(max(p.x, max(q.y, q.z)), 0.0),
            length(max(daxa_f32vec3(q.x, p.y, q.z), 0.0)) + min(max(q.x, max(p.y, q.z)), 0.0)),
        length(max(daxa_f32vec3(q.x, q.y, p.z), 0.0)) + min(max(q.x, max(q.y, p.z)), 0.0));
}
daxa_f32 sd_box_frame(in daxa_f32vec3 p, in BoundingBox box, in daxa_f32 e) {
    return sd_box_frame(p - (box.bound_max + box.bound_min) * 0.5, (box.bound_max - box.bound_min) * 0.5, e);
}
daxa_f32 sd_cylinder(in daxa_f32vec3 p, in daxa_f32 r, in daxa_f32 h) {
    daxa_f32vec2 d = abs(daxa_f32vec2(length(p.xy), p.z)) - daxa_f32vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}
daxa_f32 sd_cylinder(daxa_f32vec3 p, daxa_f32vec3 a, daxa_f32vec3 b, daxa_f32 r) {
    daxa_f32vec3 ba = b - a;
    daxa_f32vec3 pa = p - a;
    daxa_f32 baba = dot(ba, ba);
    daxa_f32 paba = dot(pa, ba);
    daxa_f32 x = length(pa * baba - ba * paba) - r * baba;
    daxa_f32 y = abs(paba - baba * 0.5) - baba * 0.5;
    daxa_f32 x2 = x * x;
    daxa_f32 y2 = y * y * baba;
    daxa_f32 d = (max(x, y) < 0.0) ? -min(x2, y2) : (((x > 0.0) ? x2 : 0.0) + ((y > 0.0) ? y2 : 0.0));
    return sign(d) * sqrt(abs(d)) / baba;
}
daxa_f32 sd_triangular_prism(in daxa_f32vec3 p, in daxa_f32 r, in daxa_f32 h) {
    const daxa_f32 k = sqrt(3.0);
    h *= 0.5 * k;
    p.xy /= h;
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0 / k;
    if (p.x + k * p.y > 0.0)
        p.xy = daxa_f32vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
    p.x -= clamp(p.x, -2.0, 0.0);
    daxa_f32 d1 = length(p.xy) * sign(-p.y) * h;
    daxa_f32 d2 = abs(p.z) - r;
    return length(max(daxa_f32vec2(d1, d2), 0.0)) + min(max(d1, d2), 0.);
}
daxa_f32 sd_hexagonal_prism(in daxa_f32vec3 p, in daxa_f32 r, in daxa_f32 h) {
    daxa_f32vec3 q = abs(p);
    const daxa_f32vec3 k = daxa_f32vec3(-0.8660254, 0.5, 0.57735);
    p = abs(p);
    p.xy -= 2.0 * min(dot(k.xy, p.xy), 0.0) * k.xy;
    daxa_f32vec2 d = daxa_f32vec2(
        length(p.xy - daxa_f32vec2(clamp(p.x, -k.z * h, k.z * h), h)) * sign(p.y - h),
        p.z - r);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}
daxa_f32 sd_octagonal_prism(in daxa_f32vec3 p, in daxa_f32 r, in daxa_f32 h) {
    const daxa_f32vec3 k = daxa_f32vec3(-0.9238795325, 0.3826834323, 0.4142135623);
    p = abs(p);
    p.xy -= 2.0 * min(dot(daxa_f32vec2(k.x, k.y), p.xy), 0.0) * daxa_f32vec2(k.x, k.y);
    p.xy -= 2.0 * min(dot(daxa_f32vec2(-k.x, k.y), p.xy), 0.0) * daxa_f32vec2(-k.x, k.y);
    p.xy -= daxa_f32vec2(clamp(p.x, -k.z * r, k.z * r), r);
    daxa_f32vec2 d = daxa_f32vec2(length(p.xy) * sign(p.y), p.z - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}
daxa_f32 sd_capsule(in daxa_f32vec3 p, in daxa_f32vec3 a, in daxa_f32vec3 b, in daxa_f32 r) {
    daxa_f32vec3 pa = p - a, ba = b - a;
    daxa_f32 h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}
daxa_f32 sd_cone(in daxa_f32vec3 p, in daxa_f32 c, in daxa_f32 h) {
    daxa_f32vec2 q = h * daxa_f32vec2(c, -1);
    daxa_f32vec2 w = daxa_f32vec2(length(p.xy), p.z);
    daxa_f32vec2 a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
    daxa_f32vec2 b = w - q * daxa_f32vec2(clamp(w.x / q.x, 0.0, 1.0), 1.0);
    daxa_f32 k = sign(q.y);
    daxa_f32 d = min(dot(a, a), dot(b, b));
    daxa_f32 s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
    return sqrt(d) * sign(s);
}
daxa_f32 sd_round_cone(in daxa_f32vec3 p, in daxa_f32 r1, in daxa_f32 r2, in daxa_f32 h) {
    daxa_f32vec2 q = daxa_f32vec2(length(p.xy), p.z);
    daxa_f32 b = (r1 - r2) / h;
    daxa_f32 a = sqrt(1.0 - b * b);
    daxa_f32 k = dot(q, daxa_f32vec2(-b, a));
    if (k < 0.0)
        return length(q) - r1;
    if (k > a * h)
        return length(q - daxa_f32vec2(0.0, h)) - r2;
    return dot(q, daxa_f32vec2(a, b)) - r1;
}
daxa_f32 sd_round_cone(in daxa_f32vec3 p, in daxa_f32vec3 a, in daxa_f32vec3 b, in daxa_f32 r1, in daxa_f32 r2) {
    daxa_f32vec3 ba = b - a;
    daxa_f32 l2 = dot(ba, ba);
    daxa_f32 rr = r1 - r2;
    daxa_f32 a2 = l2 - rr * rr;
    daxa_f32 il2 = 1.0 / l2;
    daxa_f32vec3 pa = p - a;
    daxa_f32 y = dot(pa, ba);
    daxa_f32 z = y - l2;
    daxa_f32vec3 xp = pa * l2 - ba * y;
    daxa_f32 x2 = dot(xp, xp);
    daxa_f32 y2 = y * y * l2;
    daxa_f32 z2 = z * z * l2;
    daxa_f32 k = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k)
        return sqrt(x2 + z2) * il2 - r2;
    if (sign(y) * a2 * y2 < k)
        return sqrt(x2 + y2) * il2 - r1;
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}
daxa_f32 sd_capped_cone(in daxa_f32vec3 p, in daxa_f32 r1, in daxa_f32 r2, in daxa_f32 h) {
    daxa_f32vec2 q = daxa_f32vec2(length(p.xy), p.z);
    daxa_f32vec2 k1 = daxa_f32vec2(r2, h);
    daxa_f32vec2 k2 = daxa_f32vec2(r2 - r1, 2.0 * h);
    daxa_f32vec2 ca = daxa_f32vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - h);
    daxa_f32vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / sd_shapes_dot2(k2), 0.0, 1.0);
    daxa_f32 s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s * sqrt(min(sd_shapes_dot2(ca), sd_shapes_dot2(cb)));
}
daxa_f32 sd_capped_cone(in daxa_f32vec3 p, in daxa_f32vec3 a, in daxa_f32vec3 b, in daxa_f32 ra, in daxa_f32 rb) {
    daxa_f32 rba = rb - ra;
    daxa_f32 baba = dot(b - a, b - a);
    daxa_f32 papa = dot(p - a, p - a);
    daxa_f32 paba = dot(p - a, b - a) / baba;
    daxa_f32 x = sqrt(papa - paba * paba * baba);
    daxa_f32 cax = max(0.0, x - ((paba < 0.5) ? ra : rb));
    daxa_f32 cay = abs(paba - 0.5) - 0.5;
    daxa_f32 k = rba * rba + baba;
    daxa_f32 f = clamp((rba * (x - ra) + paba * baba) / k, 0.0, 1.0);
    daxa_f32 cbx = x - ra - f * rba;
    daxa_f32 cby = paba - f;
    daxa_f32 s = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;
    return s * sqrt(min(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba));
}
daxa_f32 sd_torus(in daxa_f32vec3 p, in daxa_f32vec2 t) {
    return length(daxa_f32vec2(length(p.xy) - t.x, p.z)) - t.y;
}
daxa_f32 sd_octahedron(in daxa_f32vec3 p, in daxa_f32 s) {
    p = abs(p);
    daxa_f32 m = p.x + p.y + p.z - s;
    daxa_f32vec3 q;
    if (3.0 * p.x < m)
        q = p.xyz;
    else if (3.0 * p.y < m)
        q = p.yzx;
    else if (3.0 * p.z < m)
        q = p.zxy;
    else
        return m * 0.57735027;
    daxa_f32 k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
    return length(daxa_f32vec3(q.x, q.y - s + k, q.z - k));
}
daxa_f32 sd_pyramid(in daxa_f32vec3 p, in daxa_f32 r, in daxa_f32 h) {
    h = h / r;
    p = p / r;
    daxa_f32 m2 = h * h + 0.25;
    p.xy = abs(p.xy);
    p.xy = (p.y > p.x) ? p.yx : p.xy;
    p.xy -= 0.5;
    daxa_f32vec3 q = daxa_f32vec3(p.y, h * p.z - 0.5 * p.x, h * p.x + 0.5 * p.z);
    daxa_f32 s = max(-q.x, 0.0);
    daxa_f32 t = clamp((q.y - 0.5 * p.y) / (m2 + 0.25), 0.0, 1.0);
    daxa_f32 a = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
    daxa_f32 b = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);
    daxa_f32 d2 = min(q.y, -q.x * m2 - q.y * 0.5) > 0.0 ? 0.0 : min(a, b);
    return sqrt((d2 + q.z * q.z) / m2) * sign(max(q.z, -p.z)) * r;
}

// Random functions

float interleaved_gradient_noise(daxa_u32vec2 px) {
    return fract(52.9829189 * fract(0.06711056 * float(px.x) + 0.00583715 * float(px.y)));
}

// Jenkins hash function
daxa_u32 good_rand_hash(daxa_u32 x) {
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}
daxa_u32 good_rand_hash(daxa_u32vec2 v) { return good_rand_hash(v.x ^ good_rand_hash(v.y)); }
daxa_u32 good_rand_hash(daxa_u32vec3 v) {
    return good_rand_hash(v.x ^ good_rand_hash(v.y) ^ good_rand_hash(v.z));
}
daxa_u32 good_rand_hash(daxa_u32vec4 v) {
    return good_rand_hash(v.x ^ good_rand_hash(v.y) ^ good_rand_hash(v.z) ^ good_rand_hash(v.w));
}
daxa_f32 good_rand_float_construct(daxa_u32 m) {
    const daxa_u32 ieee_mantissa = 0x007FFFFFu;
    const daxa_u32 ieee_one = 0x3F800000u;
    m &= ieee_mantissa;
    m |= ieee_one;
    daxa_f32 f = uintBitsToFloat(m);
    return f - 1.0;
}
daxa_f32 good_rand(daxa_f32 x) { return good_rand_float_construct(good_rand_hash(floatBitsToUint(x))); }
daxa_f32 good_rand(daxa_f32vec2 v) { return good_rand_float_construct(good_rand_hash(floatBitsToUint(v))); }
daxa_f32 good_rand(daxa_f32vec3 v) { return good_rand_float_construct(good_rand_hash(floatBitsToUint(v))); }
daxa_f32 good_rand(daxa_f32vec4 v) { return good_rand_float_construct(good_rand_hash(floatBitsToUint(v))); }

// TODO: check if we need a better hash function
uint hash1(uint x) { return good_rand_hash(x); }
uint hash1_mut(inout uint h) {
    uint res = h;
    h = hash1(h);
    return res;
}
uint hash_combine2(uint x, uint y) {
    const uint M = 1664525u, C = 1013904223u;
    uint seed = (x * M + y + C) * M;
    // Tempering (from Matsumoto)
    seed ^= (seed >> 11u);
    seed ^= (seed << 7u) & 0x9d2c5680u;
    seed ^= (seed << 15u) & 0xefc60000u;
    seed ^= (seed >> 18u);
    return seed;
}
uint hash2(daxa_u32vec2 v) { return hash_combine2(v.x, hash1(v.y)); }
uint hash3(daxa_u32vec3 v) { return hash_combine2(v.x, hash2(v.yz)); }
uint hash4(daxa_u32vec4 v) { return hash_combine2(v.x, hash3(v.yzw)); }

daxa_u32 _rand_state;
void rand_seed(daxa_u32 seed) {
    _rand_state = seed;
}

daxa_f32 rand() {
    // https://www.pcg-random.org/
    _rand_state = _rand_state * 747796405u + 2891336453u;
    daxa_u32 result = ((_rand_state >> ((_rand_state >> 28u) + 4u)) ^ _rand_state) * 277803737u;
    result = (result >> 22u) ^ result;
    return result / 4294967295.0;
}

daxa_f32 rand_normal_dist() {
    daxa_f32 theta = 2.0 * PI * rand();
    daxa_f32 rho = sqrt(-2.0 * log(rand()));
    return rho * cos(theta);
}

daxa_f32vec3 rand_dir() {
    return normalize(daxa_f32vec3(
        rand_normal_dist(),
        rand_normal_dist(),
        rand_normal_dist()));
}

daxa_f32vec3 rand_hemi_dir(daxa_f32vec3 nrm) {
    daxa_f32vec3 result = rand_dir();
    return result * sign(dot(nrm, result));
}

daxa_f32vec3 rand_lambertian_nrm(daxa_f32vec3 nrm) {
    return normalize(nrm + rand_dir());
}

daxa_f32vec2 rand_circle_pt(daxa_f32vec2 random_input) {
    daxa_f32 theta = 2.0 * PI * random_input.x;
    daxa_f32 mag = sqrt(random_input.y);
    return daxa_f32vec2(cos(theta), sin(theta)) * mag;
}

daxa_f32vec2 rand_circle_pt() {
    return rand_circle_pt(daxa_f32vec2(rand(), rand()));
}

daxa_f32mat3x3 tbn_from_normal(daxa_f32vec3 nrm) {
    daxa_f32vec3 tangent = normalize(cross(nrm, -nrm.zxy));
    daxa_f32vec3 bi_tangent = cross(nrm, tangent);
    return daxa_f32mat3x3(tangent, bi_tangent, nrm);
}

// Building an Orthonormal Basis, Revisited
// http://jcgt.org/published/0006/01/01/
daxa_f32mat3x3 build_orthonormal_basis(vec3 n) {
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

    return daxa_f32mat3x3(b1, b2, n);
}

vec3 uniform_sample_cone(vec2 urand, float cos_theta_max) {
    float cos_theta = (1.0 - urand.x) + urand.x * cos_theta_max;
    float sin_theta = sqrt(clamp(1.0 - cos_theta * cos_theta, 0.0, 1.0));
    float phi = urand.y * (PI * 2.0);
    return vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

daxa_f32vec3 uniform_sample_hemisphere(daxa_f32vec2 urand) {
    float phi = urand.y * 2.0 * PI;
    float cos_theta = 1.0 - urand.x;
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    return daxa_f32vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

float rtr_encode_cos_theta_for_fp16(float x) {
    return 1 - x;
}

float rtr_decode_cos_theta_from_fp16(float x) {
    return 1 - x;
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

vec3 atmosphere(vec3 r, vec3 r0, vec3 pSun, float iSun, float rPlanet, float rAtmos, vec3 kRlh, float kMie, float shRlh, float shMie, float g) {
    // return daxa_f32vec3(0, 0, 0);
    const int iSteps = 16;
    const int jSteps = 2;
    // Normalize the sun and view directions.
    pSun = normalize(pSun);
    r = normalize(r);

    // Calculate the step size of the primary ray.
    vec2 p = rsi(r0, r, rAtmos);
    if (p.x > p.y)
        return vec3(0, 0, 0);
    p.y = min(p.y, rsi(r0, r, rPlanet).x);
    float iStepSize = (p.y - p.x) / float(iSteps);

    // Initialize the primary ray time.
    float iTime = 0.0;

    // Initialize accumulators for Rayleigh and Mie scattering.
    vec3 totalRlh = vec3(0, 0, 0);
    vec3 totalMie = vec3(0, 0, 0);

    // Initialize optical depth accumulators for the primary ray.
    float iOdRlh = 0.0;
    float iOdMie = 0.0;

    // Calculate the Rayleigh and Mie phases.
    float mu = dot(r, pSun);
    float mumu = mu * mu;
    float gg = g * g;
    float pRlh = 3.0 / (16.0 * PI) * (1.0 + mumu);
    float pMie = 3.0 / (8.0 * PI) * ((1.0 - gg) * (mumu + 1.0)) / (pow(1.0 + gg - 2.0 * mu * g, 1.5) * (2.0 + gg));

    // Sample the primary ray.
    for (int i = 0; i < iSteps; i++) {

        // Calculate the primary ray sample position.
        vec3 iPos = r0 + r * (iTime + iStepSize * 0.5);

        // Calculate the height of the sample.
        float iHeight = length(iPos) - rPlanet;

        // Calculate the optical depth of the Rayleigh and Mie scattering for this step.
        float odStepRlh = exp(-iHeight / shRlh) * iStepSize;
        float odStepMie = exp(-iHeight / shMie) * iStepSize;

        // Accumulate optical depth.
        iOdRlh += odStepRlh;
        iOdMie += odStepMie;

        // Calculate the step size of the secondary ray.
        float jStepSize = rsi(iPos, pSun, rAtmos).y / float(jSteps);

        // Initialize the secondary ray time.
        float jTime = 0.0;

        // Initialize optical depth accumulators for the secondary ray.
        float jOdRlh = 0.0;
        float jOdMie = 0.0;

        // Sample the secondary ray.
        for (int j = 0; j < jSteps; j++) {

            // Calculate the secondary ray sample position.
            vec3 jPos = iPos + pSun * (jTime + jStepSize * 0.5);

            // Calculate the height of the sample.
            float jHeight = length(jPos) - rPlanet;

            // Accumulate the optical depth.
            jOdRlh += exp(-jHeight / shRlh) * jStepSize;
            jOdMie += exp(-jHeight / shMie) * jStepSize;

            // Increment the secondary ray time.
            jTime += jStepSize;
        }

        // Calculate attenuation.
        vec3 attn = exp(-(kMie * (iOdMie + jOdMie) + kRlh * (iOdRlh + jOdRlh)));

        // Accumulate scattering.
        totalRlh += odStepRlh * attn;
        totalMie += odStepMie * attn;

        // Increment the primary ray time.
        iTime += iStepSize;
    }

    // Calculate and return the final color.
    return iSun * (pRlh * kRlh * totalRlh + pMie * kMie * totalMie);
}

bool rectangles_overlap(vec3 a_min, vec3 a_max, vec3 b_min, vec3 b_max) {
    bool x_disjoint = (a_max.x < b_min.x) || (b_max.x < a_min.x);
    bool y_disjoint = (a_max.y < b_min.y) || (b_max.y < a_min.y);
    bool z_disjoint = (a_max.z < b_min.z) || (b_max.z < a_min.z);
    return !x_disjoint && !y_disjoint && !z_disjoint;
}

// https://www.shadertoy.com/view/cdSBRG
daxa_i32 imod(daxa_i32 x, daxa_i32 m) {
    return x >= 0 ? x % m : m - 1 - (-x - 1) % m;
}
daxa_i32vec3 imod3(daxa_i32vec3 p, daxa_i32 m) {
    return daxa_i32vec3(imod(p.x, m), imod(p.y, m), imod(p.z, m));
}
daxa_i32vec3 imod3(daxa_i32vec3 p, daxa_i32vec3 m) {
    return daxa_i32vec3(imod(p.x, m.x), imod(p.y, m.y), imod(p.z, m.z));
}

daxa_f32mat4x4 rotation_matrix(daxa_f32 yaw, daxa_f32 pitch, daxa_f32 roll) {
    float sin_rot_x = sin(pitch), cos_rot_x = cos(pitch);
    float sin_rot_y = sin(roll), cos_rot_y = cos(roll);
    float sin_rot_z = sin(yaw), cos_rot_z = cos(yaw);
    return daxa_f32mat4x4(
               cos_rot_z, -sin_rot_z, 0, 0,
               sin_rot_z, cos_rot_z, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1) *
           daxa_f32mat4x4(
               1, 0, 0, 0,
               0, cos_rot_x, sin_rot_x, 0,
               0, -sin_rot_x, cos_rot_x, 0,
               0, 0, 0, 1) *
           daxa_f32mat4x4(
               cos_rot_y, -sin_rot_y, 0, 0,
               sin_rot_y, cos_rot_y, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1);
}
daxa_f32mat4x4 inv_rotation_matrix(daxa_f32 yaw, daxa_f32 pitch, daxa_f32 roll) {
    float sin_rot_x = sin(-pitch), cos_rot_x = cos(-pitch);
    float sin_rot_y = sin(-roll), cos_rot_y = cos(-roll);
    float sin_rot_z = sin(-yaw), cos_rot_z = cos(-yaw);
    return daxa_f32mat4x4(
               cos_rot_y, -sin_rot_y, 0, 0,
               sin_rot_y, cos_rot_y, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1) *
           daxa_f32mat4x4(
               1, 0, 0, 0,
               0, cos_rot_x, sin_rot_x, 0,
               0, -sin_rot_x, cos_rot_x, 0,
               0, 0, 0, 1) *
           daxa_f32mat4x4(
               cos_rot_z, -sin_rot_z, 0, 0,
               sin_rot_z, cos_rot_z, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1);
}
daxa_f32mat4x4 translation_matrix(daxa_f32vec3 pos) {
    return daxa_f32mat4x4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        pos, 1);
}

vec3 apply_inv_rotation(vec3 pt, vec3 ypr) {
    float yaw = ypr[0];
    float pitch = ypr[1];
    float roll = ypr[2];
    return (inv_rotation_matrix(ypr[0], ypr[1], ypr[2]) * vec4(pt, 0.0)).xyz;
}

daxa_f32vec3 position_world_to_clip(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec3 v) {
    daxa_f32vec4 p = (deref(globals).player.cam.world_to_view * daxa_f32vec4(v, 1));
    p = (deref(globals).player.cam.view_to_clip * p);
    return p.xyz / p.w;
}

daxa_f32vec2 get_uv(daxa_i32vec2 pix, daxa_f32vec4 tex_size) { return (daxa_f32vec2(pix) + 0.5) * tex_size.zw; }
daxa_f32vec2 get_uv(daxa_f32vec2 pix, daxa_f32vec4 tex_size) { return (pix + 0.5) * tex_size.zw; }
daxa_f32vec2 cs_to_uv(daxa_f32vec2 cs) { return cs * daxa_f32vec2(0.5, -0.5) + daxa_f32vec2(0.5, 0.5); }
daxa_f32vec2 uv_to_cs(daxa_f32vec2 uv) { return (uv - 0.5) * daxa_f32vec2(2, -2); }
daxa_f32vec2 uv_to_ss(daxa_BufferPtr(GpuInput) gpu_input, daxa_f32vec2 uv, daxa_f32vec4 tex_size) { return uv - deref(gpu_input).halton_jitter.xy * tex_size.zw * 1.0; }
daxa_f32vec2 ss_to_uv(daxa_BufferPtr(GpuInput) gpu_input, daxa_f32vec2 ss, daxa_f32vec4 tex_size) { return ss + deref(gpu_input).halton_jitter.xy * tex_size.zw * 1.0; }

struct ViewRayContext {
    daxa_f32vec4 ray_dir_cs;
    daxa_f32vec4 ray_dir_vs_h;
    daxa_f32vec4 ray_dir_ws_h;
    daxa_f32vec4 ray_origin_cs;
    daxa_f32vec4 ray_origin_vs_h;
    daxa_f32vec4 ray_origin_ws_h;
    daxa_f32vec4 ray_hit_cs;
    daxa_f32vec4 ray_hit_vs_h;
    daxa_f32vec4 ray_hit_ws_h;
};

daxa_f32vec3 ray_dir_vs(in ViewRayContext vrc) { return normalize(vrc.ray_dir_vs_h.xyz); }
daxa_f32vec3 ray_dir_ws(in ViewRayContext vrc) { return normalize(vrc.ray_dir_ws_h.xyz); }
daxa_f32vec3 ray_origin_vs(in ViewRayContext vrc) { return vrc.ray_origin_vs_h.xyz / vrc.ray_origin_vs_h.w; }
daxa_f32vec3 ray_origin_ws(in ViewRayContext vrc) { return vrc.ray_origin_ws_h.xyz / vrc.ray_origin_ws_h.w; }
daxa_f32vec3 ray_hit_vs(in ViewRayContext vrc) { return vrc.ray_hit_vs_h.xyz / vrc.ray_hit_vs_h.w; }
daxa_f32vec3 ray_hit_ws(in ViewRayContext vrc) { return vrc.ray_hit_ws_h.xyz / vrc.ray_hit_ws_h.w; }
daxa_f32vec3 biased_secondary_ray_origin_ws(in ViewRayContext vrc) {
    return ray_hit_ws(vrc) - ray_dir_ws(vrc) * (length(ray_hit_vs(vrc)) + length(ray_hit_ws(vrc))) * 1e-4;
}
daxa_f32vec3 biased_secondary_ray_origin_ws_with_normal(in ViewRayContext vrc, daxa_f32vec3 normal) {
    daxa_f32vec3 ws_abs = abs(ray_hit_ws(vrc));
    float max_comp = max(max(ws_abs.x, ws_abs.y), max(ws_abs.z, -ray_hit_vs(vrc).z));
    return ray_hit_ws(vrc) + (normal - ray_dir_ws(vrc)) * max(1e-4, max_comp * 1e-6);
}
ViewRayContext vrc_from_uv(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec2 uv) {
    ViewRayContext res;
    res.ray_dir_cs = daxa_f32vec4(uv_to_cs(uv), 0.0, 1.0);
    res.ray_dir_vs_h = deref(globals).player.cam.sample_to_view * res.ray_dir_cs;
    res.ray_dir_ws_h = deref(globals).player.cam.view_to_world * res.ray_dir_vs_h;
    res.ray_origin_cs = daxa_f32vec4(uv_to_cs(uv), 1.0, 1.0);
    res.ray_origin_vs_h = deref(globals).player.cam.sample_to_view * res.ray_origin_cs;
    res.ray_origin_ws_h = deref(globals).player.cam.view_to_world * res.ray_origin_vs_h;
    return res;
}
ViewRayContext vrc_from_uv_and_depth(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec2 uv, float depth) {
    ViewRayContext res;
    res.ray_dir_cs = daxa_f32vec4(uv_to_cs(uv), 0.0, 1.0);
    res.ray_dir_vs_h = deref(globals).player.cam.sample_to_view * res.ray_dir_cs;
    res.ray_dir_ws_h = deref(globals).player.cam.view_to_world * res.ray_dir_vs_h;
    res.ray_origin_cs = daxa_f32vec4(uv_to_cs(uv), 1.0, 1.0);
    res.ray_origin_vs_h = deref(globals).player.cam.sample_to_view * res.ray_origin_cs;
    res.ray_origin_ws_h = deref(globals).player.cam.view_to_world * res.ray_origin_vs_h;
    res.ray_hit_cs = daxa_f32vec4(uv_to_cs(uv), depth, 1.0);
    res.ray_hit_vs_h = deref(globals).player.cam.sample_to_view * res.ray_hit_cs;
    res.ray_hit_ws_h = deref(globals).player.cam.view_to_world * res.ray_hit_vs_h;
    return res;
}
#define BIAS uintBitsToFloat(0x3f800040) // uintBitsToFloat(0x3f800040) == 1.00000762939453125
ViewRayContext vrc_from_uv_and_biased_depth(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec2 uv, float depth) {
    return vrc_from_uv_and_depth(globals, uv, min(1.0, depth * BIAS));
}

daxa_f32vec3 direction_view_to_world(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec3 v) {
    return (deref(globals).player.cam.view_to_world * daxa_f32vec4(v, 0)).xyz;
}
daxa_f32vec3 direction_world_to_view(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec3 v) {
    return (deref(globals).player.cam.world_to_view * daxa_f32vec4(v, 0)).xyz;
}
daxa_f32vec3 position_world_to_view(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec3 v) {
    return (deref(globals).player.cam.world_to_view, daxa_f32vec4(v, 1)).xyz;
}
daxa_f32vec3 position_view_to_world(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec3 v) {
    return (deref(globals).player.cam.view_to_world, daxa_f32vec4(v, 1)).xyz;
}

daxa_f32vec3 position_world_to_sample(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec3 v) {
    daxa_f32vec4 p = deref(globals).player.cam.world_to_view * daxa_f32vec4(v, 1);
    p = deref(globals).player.cam.view_to_sample * p;
    return p.xyz / p.w;
}

struct Bilinear {
    daxa_f32vec2 origin;
    daxa_f32vec2 weights;
};

daxa_i32vec2 px0(in Bilinear b) { return daxa_i32vec2(b.origin); }
daxa_i32vec2 px1(in Bilinear b) { return daxa_i32vec2(b.origin) + daxa_i32vec2(1, 0); }
daxa_i32vec2 px2(in Bilinear b) { return daxa_i32vec2(b.origin) + daxa_i32vec2(0, 1); }
daxa_i32vec2 px3(in Bilinear b) { return daxa_i32vec2(b.origin) + daxa_i32vec2(1, 1); }

Bilinear get_bilinear_filter(daxa_f32vec2 uv, daxa_f32vec2 tex_size) {
    Bilinear result;
    result.origin = trunc(uv * tex_size - 0.5);
    result.weights = fract(uv * tex_size - 0.5);
    return result;
}

daxa_f32vec4 get_bilinear_custom_weights(Bilinear f, daxa_f32vec4 custom_weights) {
    daxa_f32vec4 weights;
    weights.x = (1.0 - f.weights.x) * (1.0 - f.weights.y);
    weights.y = f.weights.x * (1.0 - f.weights.y);
    weights.z = (1.0 - f.weights.x) * f.weights.y;
    weights.w = f.weights.x * f.weights.y;
    return weights * custom_weights;
}

daxa_f32vec4 apply_bilinear_custom_weights(daxa_f32vec4 s00, daxa_f32vec4 s10, daxa_f32vec4 s01, daxa_f32vec4 s11, daxa_f32vec4 w, bool should_normalize) {
    daxa_f32vec4 r = s00 * w.x + s10 * w.y + s01 * w.z + s11 * w.w;
    return r * (should_normalize ? (1.0 / dot(w, daxa_f32vec4(1.0))) : 1.0);
}

void apply_friction(daxa_BufferPtr(GpuInput) gpu_input, in out vec3 vel, vec3 friction_vec, float friction_coeff) {
    float fac = deref(gpu_input).delta_time * friction_coeff;
    vec3 new_vel = vel - normalize(friction_vec) * fac;

    if (dot(vel, new_vel) > 0.000) {
        vel = new_vel;
    } else {
        vel -= friction_vec;
    }
}

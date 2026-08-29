#ifndef UTILITIES_GPU_NORMAL_GLSL
#define UTILITIES_GPU_NORMAL_GLSL

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

vec3 u16_to_nrm(uint x) {
    return normalize(i_octahedral_16(x));
    // return i_spheremap_16(x);
}
vec3 u16_to_nrm_unnormalized(uint x) {
    return i_octahedral_16(x);
    // return i_spheremap_16(x);
}
uint nrm_to_u16(vec3 nrm) {
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

float pack_normal_11_10_11(vec3 n) {
    uint pckd = 0;
    pckd += pack_unorm(n.x * 0.5 + 0.5, 11);
    pckd += pack_unorm(n.y * 0.5 + 0.5, 10) << 11;
    pckd += pack_unorm(n.z * 0.5 + 0.5, 11) << 21;
    return uintBitsToFloat(pckd);
}

vec3 unpack_normal_11_10_11(float pckd) {
    uint p = floatBitsToUint(pckd);
    return normalize(vec3(
                         unpack_unorm(p, 11),
                         unpack_unorm(p >> 11, 10),
                         unpack_unorm(p >> 21, 11)) *
                         2.0 -
                     1.0);
}

vec2 octa_wrap(vec2 v) {
    return (1.0 - abs(v.yx)) * (step(0.0.xx, v.xy) * 2.0 - 1.0);
}

vec2 octa_encode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) {
        n.xy = octa_wrap(n.xy);
    }
    n.xy = n.xy * 0.5 + 0.5;
    return n.xy;
}

vec3 octa_decode(vec2 f) {
    f = f * 2.0 - 1.0;

    // https://twitter.com/Stubbesaurus/status/937994790553227264
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    // n.xy += select(n.xy >= 0.0, -t, t);
    n.xy -= (step(0.0, n.xy) * 2 - 1) * t;
    return normalize(n);
}

#include <renderer/kajiya/inc/math_const.glsl>

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

#endif // UTILITIES_GPU_NORMAL_GLSL

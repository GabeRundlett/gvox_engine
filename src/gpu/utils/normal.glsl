#pragma once

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

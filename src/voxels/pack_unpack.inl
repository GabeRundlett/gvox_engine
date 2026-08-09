#pragma once

#include <voxels/voxel.inl>

#if defined(__cplusplus)
#include <glm/glm.hpp>
using namespace glm;
#endif

float msign(float v) {
    return (v >= 0.0f) ? 1.0f : -1.0f;
}

vec2 map_octahedral(vec3 nor) {
    const float fac = 1.0f / (abs(nor.x) + abs(nor.y) + abs(nor.z));
    nor.x *= fac;
    nor.y *= fac;
    if (nor.z < 0.0f) {
        const vec2 temp = vec2(nor);
        nor.x = (1.0f - abs(temp.y)) * msign(temp.x);
        nor.y = (1.0f - abs(temp.x)) * msign(temp.y);
    }
    return vec2(nor.x, nor.y);
}
vec3 unmap_octahedral(vec2 v) {
    vec3 nor = vec3(v, 1.0f - abs(v.x) - abs(v.y)); // Rune Stubbe's version,
    float t = max(-nor.z, 0.0f);                    // much faster than original
    nor.x += (nor.x > 0.0f) ? -t : t;               // implementation of this
    nor.y += (nor.y > 0.0f) ? -t : t;               // technique
    return normalize(nor);
}

#define SNORM_SCALE(N) (float((1 << (N - 1u))) - 0.5f)
#define PACK_SNORM_X2(v, N)                                      \
    uvec2 d = uvec2(round(SNORM_SCALE(N) + v * SNORM_SCALE(N))); \
    return d.x | (d.y << N)
#define UNPACK_SNORM_X2(d, N) return vec2(uvec2((d), (d) >> N) & ((1u << N) - 1u)) / SNORM_SCALE(N) - 1.0f

#define UNORM_SCALE(N) (float(1 << (N)) - 1.0f)
#define PACK_UNORM(x, N) uint(round((x) * UNORM_SCALE((N))))
#define UNPACK_UNORM(x, N) (float((x) & ((1u << (N)) - 1u)) / UNORM_SCALE((N)))

#if defined(__cplusplus)
float sRGB_OETF(float a) {
    if (.0031308f >= a)
        return 12.92f * a;
    return 1.055f * pow(a, .4166666666666667f) - .055f;
}
vec3 sRGB_OETF(vec3 a) {
    return vec3(sRGB_OETF(a.r), sRGB_OETF(a.g), sRGB_OETF(a.b));
}
float sRGB_EOTF(float a) {
    if (.04045f < a)
        return pow((a + .055f) / 1.055f, 2.4f);
    return a / 12.92f;
}
vec3 sRGB_EOTF(vec3 a) {
    return vec3(sRGB_EOTF(a.r), sRGB_EOTF(a.g), sRGB_EOTF(a.b));
}
#else
#include <kajiya/inc/color/srgb.glsl>
#endif

uint pack_snorm_2x04(vec2 v) { PACK_SNORM_X2(v, 4); }
uint pack_snorm_2x08(vec2 v) { PACK_SNORM_X2(v, 8); }
uint pack_snorm_2x12(vec2 v) { PACK_SNORM_X2(v, 12); }
uint pack_snorm_2x16(vec2 v) { PACK_SNORM_X2(v, 16); }
vec2 unpack_snorm_2x04(uint d) { UNPACK_SNORM_X2(d, 4); }
vec2 unpack_snorm_2x08(uint d) { UNPACK_SNORM_X2(d, 8); }
vec2 unpack_snorm_2x12(uint d) { UNPACK_SNORM_X2(d, 12); }
vec2 unpack_snorm_2x16(uint d) { UNPACK_SNORM_X2(d, 16); }

uint pack_octahedral_08(vec3 nor) { return pack_snorm_2x04(map_octahedral(nor)); }
uint pack_octahedral_16(vec3 nor) { return pack_snorm_2x08(map_octahedral(nor)); }
uint pack_octahedral_24(vec3 nor) { return pack_snorm_2x12(map_octahedral(nor)); }
uint pack_octahedral_32(vec3 nor) { return pack_snorm_2x16(map_octahedral(nor)); }
vec3 unpack_octahedral_08(uint data) { return unmap_octahedral(unpack_snorm_2x04(data)); }
vec3 unpack_octahedral_16(uint data) { return unmap_octahedral(unpack_snorm_2x08(data)); }
vec3 unpack_octahedral_24(uint data) { return unmap_octahedral(unpack_snorm_2x12(data)); }
vec3 unpack_octahedral_32(uint data) { return unmap_octahedral(unpack_snorm_2x16(data)); }

uint pack_rgb565(vec3 col) { return (PACK_UNORM(col.r, 5) << 0) | (PACK_UNORM(col.g, 6) << 5) | (PACK_UNORM(col.b, 5) << 11); }
vec3 unpack_rgb565(uint data) { return vec3(UNPACK_UNORM(data >> 0, 5), UNPACK_UNORM(data >> 5, 6), UNPACK_UNORM(data >> 11, 5)); }

PackedVoxel pack_voxel(Voxel v) {
    return PackedVoxel(
        (pack_rgb565(sRGB_OETF(vec3(v.albedo.x, v.albedo.y, v.albedo.z)))) |
        (pack_octahedral_08(vec3(v.normal.x, v.normal.y, v.normal.z)) << 16) |
        (PACK_UNORM(sqrt(v.roughness), 4) << 24) |
        (v.material_type & 0xf) << 28);
}
Voxel unpack_voxel(PackedVoxel v) {
    vec3 col = sRGB_EOTF(unpack_rgb565(uint(v.data >> 0)));
    vec3 nrm = unpack_octahedral_08(uint(v.data >> 16));
    float roughness = UNPACK_UNORM((v.data >> 24) & 0xf, 4);
    roughness = roughness * roughness;
    uint material_type = uint(v.data >> 28) & 0xf;
    return Voxel(daxa_f32vec3(col.x, col.y, col.z), daxa_f32vec3(nrm.x, nrm.y, nrm.z), roughness, material_type);
}

#undef SNORM_SCALE
#undef PACK_SNORM_X2
#undef UNPACK_SNORM_X2
#undef UNORM_SCALE
#undef PACK_UNORM
#undef UNPACK_UNORM

#pragma once

#include <utilities/gpu/math.glsl>

uint pack_unit(float x, uint bit_n) {
    float scl = float(1u << bit_n) - 1.0;
    return uint(round(x * scl));
}
float unpack_unit(uint x, uint bit_n) {
    float scl = float(1u << bit_n) - 1.0;
    return float(x) / scl;
}

vec3 unpack_rgb(uint u) {
    vec3 result;
    result.r = float((u >> 0) & 0x3f) / 63.0;
    result.g = float((u >> 6) & 0x3f) / 63.0;
    result.b = float((u >> 12) & 0x3f) / 63.0;
    result = pow(result, vec3(2.2));
    // result = hsv2rgb(result);
    return result;
}
uint pack_rgb(vec3 f) {
    // f = rgb2hsv(f);
    f = pow(f, vec3(1.0 / 2.2));
    uint result = 0;
    result |= uint(clamp(f.r * 63.0, 0, 63)) << 0;
    result |= uint(clamp(f.g * 63.0, 0, 63)) << 6;
    result |= uint(clamp(f.b * 63.0, 0, 63)) << 12;
    return result;
}

PackedVoxel pack_voxel(Voxel v) {
    PackedVoxel result;

#if DITHER_NORMALS
    rand_seed(good_rand_hash(floatBitsToUint(v.normal)));
    const mat3 basis = build_orthonormal_basis(normalize(v.normal));
    v.normal = basis * uniform_sample_cone(vec2(rand(), rand()), cos(0.19 * 0.5));
#endif

    uint packed_roughness = pack_unit(sqrt(v.roughness), 4);
    uint packed_normal = octahedral_8(normalize(v.normal));
    uint packed_color = pack_rgb(v.color);

    result.data = (v.material_type) | (packed_roughness << 2) | (packed_normal << 6) | (packed_color << 14);

    return result;
}
Voxel unpack_voxel(PackedVoxel v) {
    Voxel result = Voxel(0, 0, vec3(0), vec3(0));

    result.material_type = (v.data >> 0) & 3;

    uint packed_roughness = (v.data >> 2) & 15;
    uint packed_normal = (v.data >> 6) & ((1 << 8) - 1);
    uint packed_color = (v.data >> 14);

    result.roughness = pow(unpack_unit(packed_roughness, 4), 2.0);
    result.normal = i_octahedral_8(packed_normal);
    result.color = unpack_rgb(packed_color);

    return result;
}

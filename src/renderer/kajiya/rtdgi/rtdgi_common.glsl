#pragma once

#include <utilities/gpu/normal.glsl>

vec4 decode_hit_normal_and_dot(vec4 val) {
    return vec4(val.xyz * 2 - 1, val.w);
}

vec4 encode_hit_normal_and_dot(vec4 val) {
    return vec4(val.xyz * 0.5 + 0.5, val.w);
}

struct TemporalReservoirOutput {
    float depth;
    vec3 ray_hit_offset_ws;
    float luminance;
    vec3 hit_normal_ws;
};

TemporalReservoirOutput TemporalReservoirOutput_from_raw(uvec4 raw) {
    vec4 ray_hit_offset_and_luminance = vec4(
        unpackHalf2x16(raw.y),
        unpackHalf2x16(raw.z));

    TemporalReservoirOutput res;
    res.depth = uintBitsToFloat(raw.x);
    res.ray_hit_offset_ws = ray_hit_offset_and_luminance.xyz;
    res.luminance = ray_hit_offset_and_luminance.w;
    res.hit_normal_ws = unpack_normal_11_10_11(uintBitsToFloat(raw.w));
    return res;
}

uvec4 as_raw(TemporalReservoirOutput self) {
    uvec4 raw;
    raw.x = floatBitsToUint(self.depth);
    raw.y = packHalf2x16(self.ray_hit_offset_ws.xy);
    raw.z = packHalf2x16(vec2(self.ray_hit_offset_ws.z, self.luminance));
    raw.w = floatBitsToUint(pack_normal_11_10_11(self.hit_normal_ws));
    return raw;
}

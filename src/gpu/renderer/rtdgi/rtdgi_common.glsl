#ifndef RTDGI_COMMON_HLSL
#define RTDGI_COMMON_HLSL

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

    static TemporalReservoirOutput from_raw(uvec4 raw) {
        vec4 ray_hit_offset_and_luminance = vec4(
            unpack_2x16f_uint(raw.y),
            unpack_2x16f_uint(raw.z));

        TemporalReservoirOutput res;
        res.depth = asfloat(raw.x);
        res.ray_hit_offset_ws = ray_hit_offset_and_luminance.xyz;
        res.luminance = ray_hit_offset_and_luminance.w;
        res.hit_normal_ws = unpack_normal_11_10_11(asfloat(raw.w));
        return res;
    }

    uvec4 as_raw() {
        uvec4 raw;
        raw.x = asuint(depth);
        raw.y = pack_2x16f_uint(ray_hit_offset_ws.xy);
        raw.z = pack_2x16f_uint(vec2(ray_hit_offset_ws.z, luminance));
        raw.w = asuint(pack_normal_11_10_11(hit_normal_ws));
        return raw;
    }
};

#endif  // RTDGI_COMMON_HLSL

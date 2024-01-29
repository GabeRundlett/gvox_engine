#pragma once

#include <utils/math_const.glsl>

#define select(cond, a, b) mix(b, a, cond)
// ((cond) ? (a) : (b))
#define rcp(x) (1.0 / x)

float square(float x) { return x * x; }
vec2 square(vec2 x) { return x * x; }
vec3 square(vec3 x) { return x * x; }
vec4 square(vec4 x) { return x * x; }

float length_squared(vec2 v) { return dot(v, v); }
float length_squared(vec3 v) { return dot(v, v); }
float length_squared(vec4 v) { return dot(v, v); }

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
    return d * M_PI / 180.0;
}
daxa_f32 rad2deg(daxa_f32 r) {
    return r * 180.0 / M_PI;
}

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

float saturate(float x) { return clamp(x, 0.0, 1.0); }
vec2 saturate(vec2 x) { return clamp(x, vec2(0.0), vec2(1.0)); }
vec3 saturate(vec3 x) { return clamp(x, vec3(0.0), vec3(1.0)); }
vec4 saturate(vec4 x) { return clamp(x, vec4(0.0), vec4(1.0)); }
float nonzero_sign(float x) {
    if (x < 0.0)
        return -1.0;
    return 1.0;
}
vec2 nonzero_sign(vec2 x) {
    return vec2(nonzero_sign(x.x), nonzero_sign(x.y));
}
vec3 nonzero_sign(vec3 x) {
    return vec3(nonzero_sign(x.x), nonzero_sign(x.y), nonzero_sign(x.z));
}
vec4 nonzero_sign(vec4 x) {
    return vec4(nonzero_sign(x.x), nonzero_sign(x.y), nonzero_sign(x.z), nonzero_sign(x.w));
}
float deg2rad(float d) {
    return d * M_PI / 180.0;
}
float rad2deg(float r) {
    return r * 180.0 / M_PI;
}

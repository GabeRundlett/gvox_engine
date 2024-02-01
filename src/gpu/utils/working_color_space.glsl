#pragma once

#include <utils/color.glsl>

// ----

// Strong suppression; reduces noise in very difficult cases but introduces a lot of bias
vec4 linear_rgb_to_crunched_luma_chroma(vec4 v) {
    v.rgb = sRGB_to_YCbCr(v.rgb);
    float k = sqrt(v.x) / max(1e-8, v.x);
    return vec4(v.rgb * k, v.a);
}
vec4 crunched_luma_chroma_to_linear_rgb(vec4 v) {
    v.rgb *= v.x;
    v.rgb = YCbCr_to_sRGB(v.rgb);
    return v;
}

// ----

vec4 linear_rgb_to_crunched_rgb(vec4 v) {
    return vec4(sqrt(v.xyz), v.w);
}
vec4 crunched_rgb_to_linear_rgb(vec4 v) {
    return vec4(v.xyz * v.xyz, v.w);
}

// ----

vec4 linear_rgb_to_linear_luma_chroma(vec4 v) {
    return vec4(sRGB_to_YCbCr(v.rgb), v.a);
}
vec4 linear_luma_chroma_to_linear_rgb(vec4 v) {
    return vec4(YCbCr_to_sRGB(v.rgb), v.a);
}

// ----
// Identity transform

vec4 linear_rgb_to_linear_rgb(vec4 v) {
    return v;
}

// ----

/*
TODO: consider this.
vec4 linear_to_working(vec4 v) {
    return log(1+sqrt(v));
}
vec4 working_to_linear(vec4 v) {
    v = exp(v) - 1.0;
    return v * v;
}*/

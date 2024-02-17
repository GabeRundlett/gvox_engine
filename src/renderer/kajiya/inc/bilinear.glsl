#pragma once

struct Bilinear {
    vec2 origin;
    vec2 weights;
};

ivec2 px0(in Bilinear b) { return ivec2(b.origin); }
ivec2 px1(in Bilinear b) { return ivec2(b.origin) + ivec2(1, 0); }
ivec2 px2(in Bilinear b) { return ivec2(b.origin) + ivec2(0, 1); }
ivec2 px3(in Bilinear b) { return ivec2(b.origin) + ivec2(1, 1); }

Bilinear get_bilinear_filter(vec2 uv, vec2 tex_size) {
    Bilinear result;
    result.origin = trunc(uv * tex_size - 0.5);
    result.weights = fract(uv * tex_size - 0.5);
    return result;
}

vec4 get_bilinear_custom_weights(Bilinear f, vec4 custom_weights) {
    vec4 weights;
    weights.x = (1.0 - f.weights.x) * (1.0 - f.weights.y);
    weights.y = f.weights.x * (1.0 - f.weights.y);
    weights.z = (1.0 - f.weights.x) * f.weights.y;
    weights.w = f.weights.x * f.weights.y;
    return weights * custom_weights;
}

vec4 apply_bilinear_custom_weights(vec4 s00, vec4 s10, vec4 s01, vec4 s11, vec4 w, bool should_normalize) {
    vec4 r = s00 * w.x + s10 * w.y + s01 * w.z + s11 * w.w;
    return r * (should_normalize ? (1.0 / dot(w, vec4(1.0))) : 1.0);
}

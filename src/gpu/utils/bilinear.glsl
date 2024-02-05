#pragma once

struct Bilinear {
    daxa_f32vec2 origin;
    daxa_f32vec2 weights;
};

daxa_i32vec2 px0(in Bilinear b) { return daxa_i32vec2(b.origin); }
daxa_i32vec2 px1(in Bilinear b) { return daxa_i32vec2(b.origin) + daxa_i32vec2(1, 0); }
daxa_i32vec2 px2(in Bilinear b) { return daxa_i32vec2(b.origin) + daxa_i32vec2(0, 1); }
daxa_i32vec2 px3(in Bilinear b) { return daxa_i32vec2(b.origin) + daxa_i32vec2(1, 1); }

Bilinear get_bilinear_filter(daxa_f32vec2 uv, daxa_f32vec2 tex_size) {
    Bilinear result;
    result.origin = trunc(uv * tex_size - 0.5);
    result.weights = fract(uv * tex_size - 0.5);
    return result;
}

daxa_f32vec4 get_bilinear_custom_weights(Bilinear f, daxa_f32vec4 custom_weights) {
    daxa_f32vec4 weights;
    weights.x = (1.0 - f.weights.x) * (1.0 - f.weights.y);
    weights.y = f.weights.x * (1.0 - f.weights.y);
    weights.z = (1.0 - f.weights.x) * f.weights.y;
    weights.w = f.weights.x * f.weights.y;
    return weights * custom_weights;
}

daxa_f32vec4 apply_bilinear_custom_weights(daxa_f32vec4 s00, daxa_f32vec4 s10, daxa_f32vec4 s01, daxa_f32vec4 s11, daxa_f32vec4 w, bool should_normalize) {
    daxa_f32vec4 r = s00 * w.x + s10 * w.y + s01 * w.z + s11 * w.w;
    return r * (should_normalize ? (1.0 / dot(w, daxa_f32vec4(1.0))) : 1.0);
}

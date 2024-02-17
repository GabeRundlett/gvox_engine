#pragma once

#include <utilities/gpu/random.glsl>

float radical_inverse_vdc(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 hammersley(uint i, uint n) {
    return vec2(float(i + 1) / n, radical_inverse_vdc(i + 1));
}

vec2 r2_sequence(uint i) {
    const float a1 = 1.0 / M_PLASTIC;
    const float a2 = 1.0 / (M_PLASTIC * M_PLASTIC);

    return fract(vec2(a1, a2) * i + 0.5);
}

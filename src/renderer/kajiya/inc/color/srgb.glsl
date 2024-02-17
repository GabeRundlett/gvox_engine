#pragma once

#include <utilities/gpu/common.glsl>

float sRGB_to_luminance(vec3 col) {
    return dot(col, vec3(0.2126, 0.7152, 0.0722));
}

// from Alex Tardiff: http://alextardif.com/Lightness.html
// Convert RGB with sRGB/Rec.709 primaries to CIE XYZ
// http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
vec3 sRGB_to_XYZ(vec3 color) {
    return mat3(0.4124564, 0.2126729, 0.0193339,
                0.3575761, 0.7151522, 0.1191920,
                0.1804375, 0.0721750, 0.9503041) *
           color;
}

// from Alex Tardiff: http://alextardif.com/Lightness.html
// Convert CIE XYZ to RGB with sRGB/Rec.709 primaries
// http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
vec3 XYZ_to_sRGB(vec3 color) {
    return mat3(
               +3.2404542, -0.9692660, +0.0556434,
               -1.5371385, +1.8760108, -0.2040259,
               -0.4985314, +0.0415560, +1.0572252) *
           color;
}

float sRGB_OETF(float a) {
    return select(.0031308f >= a, 12.92f * a, 1.055f * pow(a, .4166666666666667f) - .055f);
}

vec3 sRGB_OETF(vec3 a) {
    return vec3(sRGB_OETF(a.r), sRGB_OETF(a.g), sRGB_OETF(a.b));
}

float sRGB_EOTF(float a) {
    return select(.04045f < a, pow((a + .055f) / 1.055f, 2.4f), a / 12.92f);
}

vec3 sRGB_EOTF(vec3 a) {
    return vec3(sRGB_EOTF(a.r), sRGB_EOTF(a.g), sRGB_EOTF(a.b));
}

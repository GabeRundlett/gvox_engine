#pragma once

vec4 cubic_hermite(vec4 A, vec4 B, vec4 C, vec4 D, float t) {
    float t2 = t * t;
    float t3 = t * t * t;
    vec4 a = -A / 2.0 + (3.0 * B) / 2.0 - (3.0 * C) / 2.0 + D / 2.0;
    vec4 b = A - (5.0 * B) / 2.0 + 2.0 * C - D / 2.0;
    vec4 c = -A / 2.0 + C / 2.0;
    vec4 d = B;

    return a * t3 + b * t2 + c * t + d;
}

// https://www.shadertoy.com/view/MllSzX
#define image_sample_catmull_rom_TEMPLATE(REMAP_FUNC)                                         \
    vec4 image_sample_catmull_rom_##REMAP_FUNC(daxa_ImageViewIndex img, vec2 P, vec4 img_size) { \
        vec2 pixel = P * img_size.xy + 0.5;                                                   \
        vec2 c_onePixel = img_size.zw;                                                        \
        vec2 c_twoPixels = c_onePixel * 2.0;                                                  \
        vec2 frc = fract(pixel);                                                              \
        ivec2 ipixel = ivec2(pixel) - 1;                                                      \
        vec4 C00 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(-1, -1), 0));                \
        vec4 C10 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(0, -1), 0));                 \
        vec4 C20 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(1, -1), 0));                 \
        vec4 C30 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(2, -1), 0));                 \
        vec4 C01 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(-1, 0), 0));                 \
        vec4 C11 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(0, 0), 0));                  \
        vec4 C21 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(1, 0), 0));                  \
        vec4 C31 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(2, 0), 0));                  \
        vec4 C02 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(-1, 1), 0));                 \
        vec4 C12 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(0, 1), 0));                  \
        vec4 C22 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(1, 1), 0));                  \
        vec4 C32 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(2, 1), 0));                  \
        vec4 C03 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(-1, 2), 0));                 \
        vec4 C13 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(0, 2), 0));                  \
        vec4 C23 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(1, 2), 0));                  \
        vec4 C33 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(2, 2), 0));                  \
        vec4 CP0X = cubic_hermite(C00, C10, C20, C30, frc.x);                                 \
        vec4 CP1X = cubic_hermite(C01, C11, C21, C31, frc.x);                                 \
        vec4 CP2X = cubic_hermite(C02, C12, C22, C32, frc.x);                                 \
        vec4 CP3X = cubic_hermite(C03, C13, C23, C33, frc.x);                                 \
        return cubic_hermite(CP0X, CP1X, CP2X, CP3X, frc.y);                                  \
    }

#define image_sample_catmull_rom(REMAP_FUNC) image_sample_catmull_rom_##REMAP_FUNC

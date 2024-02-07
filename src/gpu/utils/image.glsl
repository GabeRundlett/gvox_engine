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

vec4 IdentityImageRemap_remap(vec4 v) {
    return v;
}

#include <utils/safety.glsl>

// https://www.shadertoy.com/view/MllSzX
// NOTE(grundlett): Template-ized via macros. Manual instantiation is necessary
#define image_sample_catmull_rom_TEMPLATE(REMAP_FUNC)                                                                                                                 \
    vec4 image_sample_catmull_rom_##REMAP_FUNC(daxa_ImageViewIndex img, vec2 P, vec4 img_size) {                                                                      \
        vec2 pixel = P * img_size.xy + 0.5;                                                                                                                           \
        vec2 c_onePixel = img_size.zw;                                                                                                                                \
        vec2 c_twoPixels = c_onePixel * 2.0;                                                                                                                          \
        vec2 frc = fract(pixel);                                                                                                                                      \
        ivec2 ipixel = ivec2(pixel) - 1;                                                                                                                              \
        vec4 C00 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(-1, -1), 0));                                                                                        \
        vec4 C10 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(0, -1), 0));                                                                                         \
        vec4 C20 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(1, -1), 0));                                                                                         \
        vec4 C30 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(2, -1), 0));                                                                                         \
        vec4 C01 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(-1, 0), 0));                                                                                         \
        vec4 C11 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(0, 0), 0));                                                                                          \
        vec4 C21 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(1, 0), 0));                                                                                          \
        vec4 C31 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(2, 0), 0));                                                                                          \
        vec4 C02 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(-1, 1), 0));                                                                                         \
        vec4 C12 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(0, 1), 0));                                                                                          \
        vec4 C22 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(1, 1), 0));                                                                                          \
        vec4 C32 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(2, 1), 0));                                                                                          \
        vec4 C03 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(-1, 2), 0));                                                                                         \
        vec4 C13 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(0, 2), 0));                                                                                          \
        vec4 C23 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(1, 2), 0));                                                                                          \
        vec4 C33 = REMAP_FUNC(safeTexelFetch(img, ipixel + ivec2(2, 2), 0));                                                                                          \
        vec4 CP0X = cubic_hermite(C00, C10, C20, C30, frc.x);                                                                                                         \
        vec4 CP1X = cubic_hermite(C01, C11, C21, C31, frc.x);                                                                                                         \
        vec4 CP2X = cubic_hermite(C02, C12, C22, C32, frc.x);                                                                                                         \
        vec4 CP3X = cubic_hermite(C03, C13, C23, C33, frc.x);                                                                                                         \
        return cubic_hermite(CP0X, CP1X, CP2X, CP3X, frc.y);                                                                                                          \
    }                                                                                                                                                                 \
    vec4 image_sample_catmull_rom_approx_##REMAP_FUNC(in daxa_ImageViewIndex tex, in daxa_SamplerId linearSampler, in vec2 uv, in vec2 texSize, bool useCornerTaps) { \
        /* https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1 */                                                                                     \
        /* We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding */                                            \
        /* down the sample location to get the exact center of our "starting" texel. The starting texel will be at */                                                 \
        /* location [1, 1] in the grid, where [0, 0] is the top left corner. */                                                                                       \
        vec2 samplePos = uv * texSize;                                                                                                                                \
        vec2 texPos1 = floor(samplePos - 0.5f) + 0.5f;                                                                                                                \
        /* Compute the fractional offset from our starting texel to our original sample location, which we'll */                                                      \
        /* feed into the Catmull-Rom spline function to get our filter weights. */                                                                                    \
        vec2 f = samplePos - texPos1;                                                                                                                                 \
        /* Compute the Catmull-Rom weights using the fractional offset that we calculated earlier. */                                                                 \
        /* These equations are pre-expanded based on our knowledge of where the texels will be located, */                                                            \
        /* which lets us avoid having to evaluate a piece-wise function. */                                                                                           \
        vec2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));                                                                                                                \
        vec2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);                                                                                                                  \
        vec2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));                                                                                                                 \
        vec2 w3 = f * f * (-0.5f + 0.5f * f);                                                                                                                         \
        /* Work out weighting factors and sampling offsets that will let us use bilinear filtering to */                                                              \
        /* simultaneously evaluate the middle 2 samples from the 4x4 grid. */                                                                                         \
        vec2 w12 = w1 + w2;                                                                                                                                           \
        vec2 offset12 = w2 / (w1 + w2);                                                                                                                               \
        /* Compute the final UV coordinates we'll use for sampling the texture */                                                                                     \
        vec2 texPos0 = texPos1 - 1;                                                                                                                                   \
        vec2 texPos3 = texPos1 + 2;                                                                                                                                   \
        vec2 texPos12 = texPos1 + offset12;                                                                                                                           \
        texPos0 /= texSize;                                                                                                                                           \
        texPos3 /= texSize;                                                                                                                                           \
        texPos12 /= texSize;                                                                                                                                          \
        vec4 result = vec4(0.0);                                                                                                                                      \
        if (useCornerTaps) {                                                                                                                                          \
            result += REMAP_FUNC(textureLod(daxa_sampler2D(tex, linearSampler), vec2(texPos0.x, texPos0.y), 0.0f)) * w0.x * w0.y;                                     \
        }                                                                                                                                                             \
        result += REMAP_FUNC(textureLod(daxa_sampler2D(tex, linearSampler), vec2(texPos12.x, texPos0.y), 0.0f)) * w12.x * w0.y;                                       \
        if (useCornerTaps) {                                                                                                                                          \
            result += REMAP_FUNC(textureLod(daxa_sampler2D(tex, linearSampler), vec2(texPos3.x, texPos0.y), 0.0f)) * w3.x * w0.y;                                     \
        }                                                                                                                                                             \
        result += REMAP_FUNC(textureLod(daxa_sampler2D(tex, linearSampler), vec2(texPos0.x, texPos12.y), 0.0f)) * w0.x * w12.y;                                       \
        result += REMAP_FUNC(textureLod(daxa_sampler2D(tex, linearSampler), vec2(texPos12.x, texPos12.y), 0.0f)) * w12.x * w12.y;                                     \
        result += REMAP_FUNC(textureLod(daxa_sampler2D(tex, linearSampler), vec2(texPos3.x, texPos12.y), 0.0f)) * w3.x * w12.y;                                       \
        if (useCornerTaps) {                                                                                                                                          \
            result += REMAP_FUNC(textureLod(daxa_sampler2D(tex, linearSampler), vec2(texPos0.x, texPos3.y), 0.0f)) * w0.x * w3.y;                                     \
        }                                                                                                                                                             \
        result += REMAP_FUNC(textureLod(daxa_sampler2D(tex, linearSampler), vec2(texPos12.x, texPos3.y), 0.0f)) * w12.x * w3.y;                                       \
        if (useCornerTaps) {                                                                                                                                          \
            result += REMAP_FUNC(textureLod(daxa_sampler2D(tex, linearSampler), vec2(texPos3.x, texPos3.y), 0.0f)) * w3.x * w3.y;                                     \
        }                                                                                                                                                             \
        if (!useCornerTaps) {                                                                                                                                         \
            result /= (w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y);                                                                    \
        }                                                                                                                                                             \
        return result;                                                                                                                                                \
    }                                                                                                                                                                 \
    vec4 image_sample_catmull_rom_9tap_##REMAP_FUNC(in daxa_ImageViewIndex tex, in daxa_SamplerId linearSampler, in vec2 uv, in vec2 texSize) {                       \
        return image_sample_catmull_rom_approx_##REMAP_FUNC(tex, linearSampler, uv, texSize, true);                                                                   \
    }                                                                                                                                                                 \
    vec4 image_sample_catmull_rom_5tap_##REMAP_FUNC(in daxa_ImageViewIndex tex, in daxa_SamplerId linearSampler, in vec2 uv, in vec2 texSize) {                       \
        return image_sample_catmull_rom_approx_##REMAP_FUNC(tex, linearSampler, uv, texSize, false);                                                                  \
    }

#define image_sample_catmull_rom(REMAP_FUNC) image_sample_catmull_rom_##REMAP_FUNC
#define image_sample_catmull_rom_approx(REMAP_FUNC) image_sample_catmull_rom_approx_##REMAP_FUNC
#define image_sample_catmull_rom_9tap(REMAP_FUNC) image_sample_catmull_rom_9tap_##REMAP_FUNC
#define image_sample_catmull_rom_5tap(REMAP_FUNC) image_sample_catmull_rom_5tap_##REMAP_FUNC

image_sample_catmull_rom_TEMPLATE(IdentityImageRemap_remap)

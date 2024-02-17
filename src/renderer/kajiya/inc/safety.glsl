#pragma once

// TODO: Safe fetches are super slow...
#define SAFE_FETCHES 0
#define SAFE_STORES 1

vec4 safeTexelFetch(daxa_ImageViewIndex tex, ivec2 p, int lod) {
#if SAFE_FETCHES
    ivec2 size = textureSize(daxa_texture2D(tex), lod).xy;
    if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, size))) {
        return vec4(0);
    }
#endif
    return texelFetch(daxa_texture2D(tex), p, lod);
}

uvec4 safeTexelFetchU(daxa_ImageViewIndex tex, ivec2 p, int lod) {
#if SAFE_FETCHES
    ivec2 size = textureSize(daxa_utexture2D(tex), lod).xy;
    if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, size))) {
        return uvec4(0);
    }
#endif
    return texelFetch(daxa_utexture2D(tex), p, lod);
}

ivec4 safeTexelFetchI(daxa_ImageViewIndex tex, ivec2 p, int lod) {
#if SAFE_FETCHES
    ivec2 size = textureSize(daxa_itexture2D(tex), lod).xy;
    if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, size))) {
        return ivec4(0);
    }
#endif
    return texelFetch(daxa_itexture2D(tex), p, lod);
}

void safeImageStore(daxa_ImageViewIndex img, ivec2 p, vec4 val) {
#if SAFE_STORES
    ivec2 size = imageSize(daxa_image2D(img)).xy;
    if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, size))) {
        return;
    }
#endif
    imageStore(daxa_image2D(img), p, val);
}

void safeImageStoreU(daxa_ImageViewIndex img, ivec2 p, uvec4 val) {
#if SAFE_STORES
    ivec2 size = imageSize(daxa_uimage2D(img)).xy;
    if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, size))) {
        return;
    }
#endif
    imageStore(daxa_uimage2D(img), p, val);
}

void safeImageStoreI(daxa_ImageViewIndex img, ivec2 p, ivec4 val) {
#if SAFE_STORES
    ivec2 size = imageSize(daxa_iimage2D(img)).xy;
    if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, size))) {
        return;
    }
#endif
    imageStore(daxa_iimage2D(img), p, val);
}

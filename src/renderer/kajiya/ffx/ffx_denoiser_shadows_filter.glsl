/**********************************************************************
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
********************************************************************/
#pragma once

#include "ffx_denoiser_shadows_util.glsl"

shared uint g_FFX_DNSR_Shadows_shared_input[16][16];
shared float g_FFX_DNSR_Shadows_shared_depth[16][16];
shared uint g_FFX_DNSR_Shadows_shared_normals_xy[16][16];
shared uint g_FFX_DNSR_Shadows_shared_normals_zw[16][16];

uint FFX_DNSR_Shadows_PackFloat16(vec2 v) {
    return packHalf2x16(v);
}

vec2 FFX_DNSR_Shadows_UnpackFloat16(uint a) {
    return unpackHalf2x16(a);
}

vec2 FFX_DNSR_Shadows_LoadInputFromGroupSharedMemory(ivec2 idx) {
    return FFX_DNSR_Shadows_UnpackFloat16(g_FFX_DNSR_Shadows_shared_input[idx.y][idx.x]);
}

float FFX_DNSR_Shadows_LoadDepthFromGroupSharedMemory(ivec2 idx) {
    return g_FFX_DNSR_Shadows_shared_depth[idx.y][idx.x];
}

vec3 FFX_DNSR_Shadows_LoadNormalsFromGroupSharedMemory(ivec2 idx) {
    vec3 normals;
    normals.xy = FFX_DNSR_Shadows_UnpackFloat16(g_FFX_DNSR_Shadows_shared_normals_xy[idx.y][idx.x]);
    normals.z = FFX_DNSR_Shadows_UnpackFloat16(g_FFX_DNSR_Shadows_shared_normals_zw[idx.y][idx.x]).x;
    return normals;
}

void FFX_DNSR_Shadows_StoreInGroupSharedMemory(ivec2 idx, vec3 normals, vec2 a_input, float depth) {
    g_FFX_DNSR_Shadows_shared_input[idx.y][idx.x] = FFX_DNSR_Shadows_PackFloat16(a_input);
    g_FFX_DNSR_Shadows_shared_depth[idx.y][idx.x] = depth;
    g_FFX_DNSR_Shadows_shared_normals_xy[idx.y][idx.x] = FFX_DNSR_Shadows_PackFloat16(normals.xy);
    g_FFX_DNSR_Shadows_shared_normals_zw[idx.y][idx.x] = FFX_DNSR_Shadows_PackFloat16(vec2(normals.z, 0));
}

void FFX_DNSR_Shadows_LoadWithOffset(ivec2 did, ivec2 offset, out vec3 normals, out vec2 a_input, out float depth) {
    did += offset;

    const ivec2 p = clamp(did, ivec2(0, 0), ivec2(FFX_DNSR_Shadows_GetBufferDimensions()) - 1);
    normals = FFX_DNSR_Shadows_ReadNormals(p);
    a_input = FFX_DNSR_Shadows_ReadInput(p);
    depth = FFX_DNSR_Shadows_ReadDepth(p);
}

void FFX_DNSR_Shadows_StoreWithOffset(ivec2 gtid, ivec2 offset, vec3 normals, vec2 a_input, float depth) {
    gtid += offset;
    FFX_DNSR_Shadows_StoreInGroupSharedMemory(gtid, normals, a_input, depth);
}

void FFX_DNSR_Shadows_InitializeGroupSharedMemory(ivec2 did, ivec2 gtid) {
    ivec2 offset_0 = ivec2(0, 0);
    ivec2 offset_1 = ivec2(8, 0);
    ivec2 offset_2 = ivec2(0, 8);
    ivec2 offset_3 = ivec2(8, 8);

    vec3 normals_0;
    vec2 input_0;
    float depth_0;

    vec3 normals_1;
    vec2 input_1;
    float depth_1;

    vec3 normals_2;
    vec2 input_2;
    float depth_2;

    vec3 normals_3;
    vec2 input_3;
    float depth_3;

    /// XA
    /// BC

    did -= 4;
    FFX_DNSR_Shadows_LoadWithOffset(did, offset_0, normals_0, input_0, depth_0); // X
    FFX_DNSR_Shadows_LoadWithOffset(did, offset_1, normals_1, input_1, depth_1); // A
    FFX_DNSR_Shadows_LoadWithOffset(did, offset_2, normals_2, input_2, depth_2); // B
    FFX_DNSR_Shadows_LoadWithOffset(did, offset_3, normals_3, input_3, depth_3); // C

    FFX_DNSR_Shadows_StoreWithOffset(gtid, offset_0, normals_0, input_0, depth_0); // X
    FFX_DNSR_Shadows_StoreWithOffset(gtid, offset_1, normals_1, input_1, depth_1); // A
    FFX_DNSR_Shadows_StoreWithOffset(gtid, offset_2, normals_2, input_2, depth_2); // B
    FFX_DNSR_Shadows_StoreWithOffset(gtid, offset_3, normals_3, input_3, depth_3); // C
}

float FFX_DNSR_Shadows_GetShadowSimilarity(float x1, float x2, float sigma) {
    return exp(-abs(x1 - x2) / sigma);
}

float FFX_DNSR_Shadows_GetDepthSimilarity(float x1, float x2, float sigma) {
    float depth_diff = 1.0 - (x1 / x2);
    return exp2(-abs(depth_diff) / sigma);
}

float FFX_DNSR_Shadows_GetNormalSimilarity(vec3 x1, vec3 x2) {
    return pow(clamp(dot(x1, x2), 0, 1), 32.0f);
}

float FFX_DNSR_Shadows_FetchFilteredVarianceFromGroupSharedMemory(ivec2 pos) {
#if 0
        const int k = 1;
        float variance = 0.0f;
        const float kernel[2][2] =
        {
            { 1.0f / 4.0f, 1.0f / 8.0f  },
            { 1.0f / 8.0f, 1.0f / 16.0f }
        };
        for (int y = -k; y <= k; ++y)
        {
            for (int x = -k; x <= k; ++x)
            {
                const float w = kernel[abs(x)][abs(y)];
                variance += w * FFX_DNSR_Shadows_LoadInputFromGroupSharedMemory(pos + ivec2(x, y)).y;
            }
        }
        return variance;
#else
    const int k = 1;
    float variance = FFX_DNSR_Shadows_LoadInputFromGroupSharedMemory(pos).y;
    /*for (int y = -k; y <= k; ++y)
    {
        for (int x = -k; x <= k; ++x)
        {
            variance = min(
                variance,
                10 * FFX_DNSR_Shadows_LoadInputFromGroupSharedMemory(pos + ivec2(x, y)
            ).y);
        }
    }*/
    return variance;
#endif
}

void FFX_DNSR_Shadows_DenoiseFromGroupSharedMemory(uvec2 did, uvec2 gtid, inout float weight_sum, inout vec2 shadow_sum, float depth, uint stepsize) {
    // Load our center sample
    const vec2 shadow_center = FFX_DNSR_Shadows_LoadInputFromGroupSharedMemory(ivec2(gtid));
    const vec3 normal_center = FFX_DNSR_Shadows_LoadNormalsFromGroupSharedMemory(ivec2(gtid));

    weight_sum = 1.0f;
    shadow_sum = shadow_center;

    const float variance = FFX_DNSR_Shadows_FetchFilteredVarianceFromGroupSharedMemory(ivec2(gtid));
    const float std_deviation = sqrt(max(variance + 1e-9f, 0.0f));
    const float depth_center = depth; // linearize the depth value

    // Iterate filter kernel
    const int k = 1;

    // Narrow down the kernel when variance is already low.
    // Very ad-hoc; helps with thin/small shadow casters.
    // const float kernel_sharpening = max(1e-10, 1.0 - exp(-3.0 * std_deviation));
    const float kernel_sharpening = max(1e-10, 1.0 - max(0.0, 1.0 - 2.0 * std_deviation) * max(0.0, 1.0 - 2.0 * std_deviation));
    // const float kernel_sharpening = 1;
    const float kernel[3] = {
        1.0f,
        exp2(-0.5849625007211563 / kernel_sharpening), // 2/3
        exp2(-2.584962500721156 / kernel_sharpening),  // 1/6
    };

    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            // Should we process this sample?
            const ivec2 step = ivec2(x, y) * int(stepsize);
            const ivec2 gtid_idx = ivec2(gtid) + step;
            const ivec2 did_idx = ivec2(did) + step;

            float depth_neigh = FFX_DNSR_Shadows_LoadDepthFromGroupSharedMemory(gtid_idx);
            vec3 normal_neigh = FFX_DNSR_Shadows_LoadNormalsFromGroupSharedMemory(gtid_idx);
            vec2 shadow_neigh = FFX_DNSR_Shadows_LoadInputFromGroupSharedMemory(gtid_idx);

            float sky_pixel_multiplier = select(((x == 0 && y == 0) || depth_neigh >= 1.0f || depth_neigh <= 0.0f), 0, 1); // Zero weight for sky pixels

            // Evaluate the edge-stopping function
            float w = kernel[abs(x)] * kernel[abs(y)]; // kernel weight
            w *= FFX_DNSR_Shadows_GetShadowSimilarity(shadow_center.x, shadow_neigh.x, std_deviation);
            w *= FFX_DNSR_Shadows_GetDepthSimilarity(depth_center, depth_neigh, FFX_DNSR_Shadows_GetDepthSimilaritySigma());
            w *= FFX_DNSR_Shadows_GetNormalSimilarity(normal_center, normal_neigh);
            w *= sky_pixel_multiplier;

            // Accumulate the filtered sample
            shadow_sum += vec2(w, w * w) * shadow_neigh;
            weight_sum += w;
        }
    }
}

vec2 FFX_DNSR_Shadows_ApplyFilterWithPrecache(uvec2 did, uvec2 gtid, uint stepsize) {
    float weight_sum = 1.0;
    vec2 shadow_sum = vec2(0.0);

    FFX_DNSR_Shadows_InitializeGroupSharedMemory(ivec2(did), ivec2(gtid));
    bool needs_denoiser = FFX_DNSR_Shadows_IsShadowReciever(did);
    barrier();
    if (needs_denoiser) {
        float depth = FFX_DNSR_Shadows_ReadDepth(did);
        gtid += 4; // Center threads in groupshared memory
        FFX_DNSR_Shadows_DenoiseFromGroupSharedMemory(did, gtid, weight_sum, shadow_sum, depth, stepsize);
    }

    float mean = shadow_sum.x / weight_sum;
    float variance = shadow_sum.y / (weight_sum * weight_sum);
    return vec2(mean, variance);
}

void FFX_DNSR_Shadows_ReadTileMetaData(uvec2 gid, out bool is_cleared, out bool all_in_light) {
    uint meta_data = FFX_DNSR_Shadows_ReadTileMetaData(gid.y * FFX_DNSR_Shadows_RoundedDivide(FFX_DNSR_Shadows_GetBufferDimensions().x, 8) + gid.x);
    is_cleared = (meta_data & TILE_META_DATA_CLEAR_MASK) != 0;
    all_in_light = (meta_data & TILE_META_DATA_LIGHT_MASK) != 0;
}

vec2 FFX_DNSR_Shadows_FilterSoftShadowsPass(uvec2 gid, uvec2 gtid, uvec2 did, out bool bWriteResults, const uint pass, const uint stepsize) {
    bool is_cleared;
    bool all_in_light;
    FFX_DNSR_Shadows_ReadTileMetaData(gid, is_cleared, all_in_light);

    bWriteResults = false;
    vec2 results = vec2(0, 0);
    // [branch]
    if (is_cleared) {
        if (pass != 1) {
            results.x = select(all_in_light, 1.0, 0.0);
            bWriteResults = true;
        }
    } else {
        results = FFX_DNSR_Shadows_ApplyFilterWithPrecache(did, gtid, stepsize);
        bWriteResults = true;
    }

    return results;
}

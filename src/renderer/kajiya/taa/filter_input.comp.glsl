#include <renderer/kajiya/taa.inl>

#include <g_samplers>
#include "../inc/camera.glsl"
#include "../inc/color.glsl"
#include "taa_common.glsl"

#include "../inc/safety.glsl"

DAXA_DECL_PUSH_CONSTANT(TaaFilterInputComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewIndex input_image = push.uses.input_image;
daxa_ImageViewIndex depth_image = push.uses.depth_image;
daxa_ImageViewIndex filtered_input_img = push.uses.filtered_input_img;
daxa_ImageViewIndex filtered_input_deviation_img = push.uses.filtered_input_deviation_img;

vec4 InputRemap_remap(vec4 v) {
    return vec4(sRGB_to_YCbCr(decode_rgb(v.rgb)), 1);
}

struct FilteredInput {
    vec3 clamped_ex;
    vec3 var;
};

float fetch_depth(ivec2 px) {
    return safeTexelFetch(depth_image, px, 0).r;
}
vec4 fetch_input(ivec2 px) {
    return safeTexelFetch(input_image, px, 0);
}

FilteredInput filter_input_inner(uvec2 px, float center_depth, float luma_cutoff, float depth_scale) {
    vec3 iex = vec3(0);
    vec3 iex2 = vec3(0);
    float iwsum = 0;

    vec3 clamped_iex = vec3(0);
    float clamped_iwsum = 0;

    const int k = 1;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            const ivec2 spx_offset = ivec2(x, y);
            const float distance_w = exp(-(0.8 / (k * k)) * dot(vec2(spx_offset), vec2(spx_offset)));

            const ivec2 spx = ivec2(px) + spx_offset;
            vec3 s = InputRemap_remap(fetch_input(spx)).rgb;

            const float depth = fetch_depth(spx);
            float w = 1;
            w *= exp2(-min(16, depth_scale * inverse_depth_relative_diff(center_depth, depth)));
            w *= distance_w;
            w *= pow(clamp(luma_cutoff / s.x, 0.0, 1.0), 8);

            clamped_iwsum += w;
            clamped_iex += s * w;

            iwsum += 1;
            iex += s;
            iex2 += s * s;
        }
    }

    clamped_iex /= clamped_iwsum;

    iex /= iwsum;
    iex2 /= iwsum;

    FilteredInput res;
    res.clamped_ex = clamped_iex;
    res.var = max(vec3(0.0), iex2 - iex * iex);

    return res;
}

layout(local_size_x = TAA_WG_SIZE_X, local_size_y = TAA_WG_SIZE_Y, local_size_z = 1) in;
void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    const float center_depth = fetch_depth(px);

    // Filter the input, with a cross-bilateral weight based on depth
    FilteredInput filtered_input = filter_input_inner(px, center_depth, 1e10, 200);

    // Filter the input again, but add another cross-bilateral weight, reducing the weight of
    // inputs brighter than the just-estimated luminance mean. This clamps bright outliers in the input.
    FilteredInput clamped_filtered_input = filter_input_inner(px, center_depth, filtered_input.clamped_ex.x * 1.001, 200);

    safeImageStore(filtered_input_img, px, vec4(clamped_filtered_input.clamped_ex, 0.0));
    safeImageStore(filtered_input_deviation_img, px, vec4(sqrt(filtered_input.var), 0.0));
}

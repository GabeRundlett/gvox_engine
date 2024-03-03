#include <renderer/kajiya/taa.inl>

#include <g_samplers>
#include "../inc/camera.glsl"
#include "../inc/color.glsl"
#include "taa_common.glsl"

#include "../inc/safety.glsl"

DAXA_DECL_PUSH_CONSTANT(TaaFilterHistoryComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewIndex reprojected_history_img = push.uses.reprojected_history_img;
daxa_ImageViewIndex filtered_history_img = push.uses.filtered_history_img;

vec4 fetch_input(ivec2 px) {
    return safeTexelFetch(reprojected_history_img, px, 0);
}

vec3 filter_input(vec2 uv, float luma_cutoff, int kernel_radius) {
    vec3 iex = vec3(0);
    float iwsum = 0;

    // Note: + epislon to counter precision loss, which manifests itself
    // as bad rounding in a 2x upscale, showing stair-stepping artifacts.
    ivec2 src_px = ivec2(floor(uv * push.input_tex_size.xy + 1e-3));

    const int k = kernel_radius;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            const ivec2 spx_offset = ivec2(x, y);
            const ivec2 spx = ivec2(src_px) + spx_offset;

            // TODO: consider a weight based on uv diffs between the low-res
            // output `uv` and the low-res input `spx`.
            const float distance_w = exp(-(0.8 / (k * k)) * dot(vec2(spx_offset), vec2(spx_offset)));

            vec3 s = sRGB_to_YCbCr(fetch_input(spx).rgb);

            float w = 1;
            w *= distance_w;
            w *= pow(clamp(luma_cutoff / s.x, 0.0, 1.0), 8);

            iwsum += w;
            iex += s * w;
        }
    }

    return iex / iwsum;
}

void filter_history(uvec2 px, int kernel_radius) {
    vec2 uv = get_uv(px, vec4(push.output_tex_size.xy, 1.0 / push.output_tex_size.xy));
    float filtered_luma = filter_input(uv, 1e10, kernel_radius).x;
    safeImageStore(filtered_history_img, ivec2(px), vec4(filter_input(uv, filtered_luma * 1.001, kernel_radius), 0.0));
}

layout(local_size_x = TAA_WG_SIZE_X, local_size_y = TAA_WG_SIZE_Y, local_size_z = 1) in;
void main() {
    uvec2 px = gl_GlobalInvocationID.xy;
    if (push.input_tex_size.x / push.output_tex_size.x > 1.75) {
        // If we're upscaling, history is at a higher resolution than
        // the new frame, so we need to filter history more.
        filter_history(px, 2);
    } else {
        filter_history(px, 1);
    }
}

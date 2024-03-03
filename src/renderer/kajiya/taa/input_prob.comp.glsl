#include <renderer/kajiya/taa.inl>

#include <g_samplers>
#include "../inc/camera.glsl"
#include "../inc/color.glsl"
#include "../inc/image.glsl"
#include "taa_common.glsl"

DAXA_DECL_PUSH_CONSTANT(TaaInputProbComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewIndex input_image = push.uses.input_image;
daxa_ImageViewIndex filtered_input_img = push.uses.filtered_input_img;
daxa_ImageViewIndex filtered_input_deviation_img = push.uses.filtered_input_deviation_img;
daxa_ImageViewIndex reprojected_history_img = push.uses.reprojected_history_img;
daxa_ImageViewIndex filtered_history_img = push.uses.filtered_history_img;
daxa_ImageViewIndex reprojection_map = push.uses.reprojection_map;
daxa_ImageViewIndex depth_image = push.uses.depth_image;
daxa_ImageViewIndex smooth_var_history_tex = push.uses.smooth_var_history_tex;
daxa_ImageViewIndex velocity_history_tex = push.uses.velocity_history_tex;
daxa_ImageViewIndex input_prob_img = push.uses.input_prob_img;

vec4 InputRemap_remap(vec4 v) {
    return vec4(sRGB_to_YCbCr(decode_rgb(v.rgb)), 1);
}

vec4 HistoryRemap_remap(vec4 v) {
    return vec4(sRGB_to_YCbCr(v.rgb), 1);
}

vec4 fetch_filtered_input(ivec2 px) {
    return safeTexelFetch(filtered_input_img, px, 0);
}

vec3 fetch_filtered_input_dev(ivec2 px) {
    return safeTexelFetch(filtered_input_deviation_img, px, 0).rgb;
}

vec4 fetch_reproj(ivec2 px) {
    return safeTexelFetch(reprojection_map, px, 0);
}

#define PREFETCH_RADIUS 2
#define PREFETCH_GROUP_SIZE 16

struct FetchResult {
    vec3 filtered_input;
    vec2 vel;
};

FetchResult do_fetch(ivec2 px) {
    const vec3 s = fetch_filtered_input(px).rgb;
    const vec2 vel = fetch_reproj(px).xy;
    return FetchResult(s, vel);
}

#include "../inc/prefetch.glsl"

layout(local_size_x = PREFETCH_GROUP_SIZE, local_size_y = PREFETCH_GROUP_SIZE, local_size_z = 1) in;
void main() {
    float input_prob = 0;
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);

    {
        // InputRemap input_remap = InputRemap::create();

        // Estimate input variance from a pretty large spatial neighborhood
        // We'll combine it with a temporally-filtered variance estimate later.
        vec3 ivar = vec3(0);
        {
            const int k = 1;
            for (int y = -k; y <= k; ++y) {
                for (int x = -k; x <= k; ++x) {
                    ivar = max(ivar, fetch_filtered_input_dev(px + ivec2(x, y) * 2));
                }
            }
            ivar = square(ivar);
        }

        const vec2 input_uv = (px + vec2(deref(gpu_input).halton_jitter)) / push.input_tex_size.xy;

        const vec4 closest_history = textureLod(daxa_sampler2D(filtered_history_img, g_sampler_nnc), input_uv, 0);
        const vec3 closest_smooth_var = textureLod(daxa_sampler2D(smooth_var_history_tex, g_sampler_lnc), input_uv + fetch_reproj(px).xy, 0).rgb;
        const vec2 closest_vel = textureLod(daxa_sampler2D(velocity_history_tex, g_sampler_lnc), input_uv + fetch_reproj(px).xy, 0).xy * deref(gpu_input).delta_time;

        // Combine spaital and temporla variance. We generally want to use
        // the smoothed temporal estimate, but bound it by this frame's input,
        // to quickly react for large-scale temporal changes.
        const vec3 combined_var = min(closest_smooth_var, ivar * 10);
        // const vec3 combined_var = closest_smooth_var;
        // const vec3 combined_var = ivar;

        // Check this frame's input, and see how closely it resembles history,
        // taking the variance estimate into account.
        //
        // The idea here is that the new frames are samples from an unknown
        // function, while `closest_history` and `combined_var` are estimates
        // of its mean and variance. We find the probability that the given sample
        // belongs to the estimated distribution.
        //
        // Use a small neighborhood search, because the input may
        // flicker from frame to frame due to the temporal jitter.

        do_prefetch();

        {
            int k = 1;
            for (int y = -k; y <= k; ++y) {
                for (int x = -k; x <= k; ++x) {
                    FetchResult fetch_result = prefetch_tap(ivec2(px + ivec2(x, y)));

                    const vec3 s = fetch_result.filtered_input;
                    const vec3 idiff = s - closest_history.rgb;

                    const vec2 vel = fetch_result.vel;
                    const float vdiff = length((vel - closest_vel) / max(vec2(1.0), abs(vel + closest_vel)));

                    float prob = exp2(-1.0 * length(idiff * idiff / max(vec3(1e-6), combined_var)) - 1000 * vdiff);

                    input_prob = max(input_prob, prob);
                }
            }
        }
    }

    safeImageStore(input_prob_img, ivec2(px), vec4(input_prob, 0, 0, 0));
}

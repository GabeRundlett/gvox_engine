#include "../inc/samplers.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/color.hlsl"
#include "../inc/image.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/unjitter_taa.hlsl"
#include "../inc/soft_color_clamp.hlsl"
#include "taa_common.hlsl"

[[vk::binding(0)]] Texture2D<vec4> input_tex;
[[vk::binding(1)]] Texture2D<vec4> filtered_input_tex;
[[vk::binding(2)]] Texture2D<vec3> filtered_input_dev_tex;
[[vk::binding(3)]] Texture2D<vec4> history_tex;
[[vk::binding(4)]] Texture2D<vec4> filtered_history_tex;
[[vk::binding(5)]] Texture2D<vec4> reprojection_tex;
[[vk::binding(6)]] Texture2D<float> depth_tex;
[[vk::binding(7)]] Texture2D<vec3> smooth_var_history_tex;
[[vk::binding(8)]] Texture2D<vec2> velocity_history_tex;
[[vk::binding(9)]] RWTexture2D<float> output_tex;
[[vk::binding(10)]] cbuffer _ {
    vec4 input_tex_size;
};

struct InputRemap {
    static InputRemap create() {
        InputRemap res;
        return res;
    }

    vec4 remap(vec4 v) {
        return vec4(sRGB_to_YCbCr(decode_rgb(v.rgb)), 1);
    }
};

struct HistoryRemap {
    static HistoryRemap create() {
        HistoryRemap res;
        return res;
    }

    vec4 remap(vec4 v) {
        return vec4(sRGB_to_YCbCr(v.rgb), 1);
    }
};

[numthreads(8, 8, 1)]
void main(uvec2 px: SV_DispatchThreadID) {
    float input_prob = 0;

    {
        InputRemap input_remap = InputRemap::create();

        // Estimate input variance from a pretty large spatial neighborhood
        // We'll combine it with a temporally-filtered variance estimate later.
        vec3 ivar = 0;
        {
            const int k = 1;
            for (int y = -k; y <= k; ++y) {
                for (int x = -k; x <= k; ++x) {
                    ivar = max(ivar, filtered_input_dev_tex[px + ivec2(x, y) * 2]);
                }
            }
            ivar = square(ivar);
        }

        const vec2 input_uv = (px + frame_constants.view_constants.sample_offset_pixels) * input_tex_size.zw;

        const vec4 closest_history = filtered_history_tex.SampleLevel(sampler_nnc, input_uv, 0);
        const vec3 closest_smooth_var = smooth_var_history_tex.SampleLevel(sampler_lnc, input_uv + reprojection_tex[px].xy, 0);
        const vec2 closest_vel = velocity_history_tex.SampleLevel(sampler_lnc, input_uv + reprojection_tex[px].xy, 0).xy * frame_constants.delta_time_seconds;

        // Combine spaital and temporla variance. We generally want to use
        // the smoothed temporal estimate, but bound it by this frame's input,
        // to quickly react for large-scale temporal changes.
        const vec3 combined_var = min(closest_smooth_var, ivar * 10);
        //const vec3 combined_var = closest_smooth_var;
        //const vec3 combined_var = ivar;

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
        {
            int k = 1;
            for (int y = -k; y <= k; ++y) {
                for (int x = -k; x <= k; ++x) {
                    const vec3 s = filtered_input_tex[px + ivec2(x, y)].rgb;
                    const vec3 idiff = s - closest_history.rgb;

                    const vec2 vel = reprojection_tex[px + ivec2(x, y)].xy;
                    const float vdiff = length((vel - closest_vel) / max(1, abs(vel + closest_vel)));

                    float prob = exp2(-1.0 * length(idiff * idiff / max(1e-6, combined_var)) - 1000 * vdiff);

                    input_prob = max(input_prob, prob);
                }
            }
        }
    }

    output_tex[px] = input_prob;
}

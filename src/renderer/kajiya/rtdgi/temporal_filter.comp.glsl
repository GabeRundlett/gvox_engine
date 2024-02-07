#include <renderer/kajiya/rtdgi.inl>

#include <utils/math.glsl>
#include <utils/camera.glsl>
// #include <utils/samplers.glsl>
#include <utils/color.glsl>
// #include <utils/uv.glsl>
// #include <utils/frame_constants.glsl>
// #include <utils/soft_color_clamp.glsl>
#include <utils/working_color_space.glsl>
#include <utils/safety.glsl>

DAXA_DECL_PUSH_CONSTANT(RtdgiTemporalFilterComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewIndex input_tex = push.uses.input_tex;
daxa_ImageViewIndex history_tex = push.uses.history_tex;
daxa_ImageViewIndex variance_history_tex = push.uses.variance_history_tex;
daxa_ImageViewIndex reprojection_tex = push.uses.reprojection_tex;
daxa_ImageViewIndex rt_history_invalidity_tex = push.uses.rt_history_invalidity_tex;
daxa_ImageViewIndex output_tex = push.uses.output_tex;
daxa_ImageViewIndex history_output_tex = push.uses.history_output_tex;
daxa_ImageViewIndex variance_history_output_tex = push.uses.variance_history_output_tex;

#define USE_BBOX_CLAMP 1

#if 0
    // Linear accumulation, for comparisons with path tracing
    vec4 pass_through(vec4 v) { return v; }
#define linear_to_working pass_through
#define working_to_linear pass_through
    float working_luma(vec3 v) { return sRGB_to_luminance(v); }
#else
#define linear_to_working linear_rgb_to_crunched_luma_chroma
#define working_to_linear crunched_luma_chroma_to_linear_rgb
float working_luma(vec3 v) { return v.x; }
#endif

#define USE_TEMPORAL_FILTER 1

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    uvec2 px = gl_GlobalInvocationID.xy;
#if USE_TEMPORAL_FILTER == 0
    safeImageStore(output_tex, ivec2(px), max(vec4(0.0), safeTexelFetch(input_tex, ivec2(px), 0)));
    safeImageStore(history_output_tex, ivec2(px), vec4(max(vec3(0.0), safeTexelFetch(input_tex, ivec2(px), 0).rgb), 32));
    return;
#endif

    vec2 uv = get_uv(px, push.output_tex_size);

    vec4 center = linear_to_working(safeTexelFetch(input_tex, ivec2(px), 0));
    vec4 reproj = safeTexelFetch(reprojection_tex, ivec2(px), 0);

    const vec4 history_mult = vec4((deref(gpu_input).pre_exposure_delta).xxx, 1);
    vec4 history = linear_to_working(safeTexelFetch(history_tex, ivec2(px), 0) * history_mult);

    // output_tex[px] = center;
    // return;

#if 1
    vec4 vsum = 0.0.xxxx;
    vec4 vsum2 = 0.0.xxxx;
    float wsum = 0.0;
    float hist_diff = 0.0;
    float hist_vsum = 0.0;
    float hist_vsum2 = 0.0;

    // float dev_sum = 0.0;

    const int k = 2;
    {
        for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                vec4 neigh = linear_to_working(safeTexelFetch(input_tex, ivec2(px + ivec2(x, y)), 0));
                vec4 hist_neigh = linear_to_working(safeTexelFetch(history_tex, ivec2(px + ivec2(x, y)), 0) * history_mult);

                float neigh_luma = working_luma(neigh.rgb);
                float hist_luma = working_luma(hist_neigh.rgb);

                float w = exp(-3.0 * float(x * x + y * y) / float((k + 1.) * (k + 1.)));
                vsum += neigh * w;
                vsum2 += neigh * neigh * w;
                wsum += w;

                // dev_sum += neigh.a * neigh.a * w;

                // hist_diff += (neigh_luma - hist_luma) * (neigh_luma - hist_luma) * w;
                hist_diff += abs(neigh_luma - hist_luma) / max(1e-5, neigh_luma + hist_luma) * w;
                hist_vsum += hist_luma * w;
                hist_vsum2 += hist_luma * hist_luma * w;
            }
        }
    }

    vec4 ex = vsum / wsum;
    vec4 ex2 = vsum2 / wsum;
    vec4 dev = sqrt(max(0.0.xxxx, ex2 - ex * ex));

    hist_diff /= wsum;
    hist_vsum /= wsum;
    hist_vsum2 /= wsum;
    // dev_sum /= wsum;

    const vec2 moments_history =
        textureLod(daxa_sampler2D(variance_history_tex, deref(gpu_input).sampler_lnc), uv + reproj.xy, 0).xy *
        vec2(deref(gpu_input).pre_exposure_delta, deref(gpu_input).pre_exposure_delta * deref(gpu_input).pre_exposure_delta);

    // const float center_luma = working_luma(center.rgb);
    const float center_luma = working_luma(center.rgb) + (hist_vsum - working_luma(ex.rgb)); // - 0.5 * working_luma(control_variate.rgb));
    const vec2 current_moments = vec2(center_luma, center_luma * center_luma);
    safeImageStore(variance_history_output_tex, ivec2(px), vec4(max(vec2(0.0), mix(moments_history, current_moments, vec2(0.25))), 0.0, 0.0));
    const float center_temporal_dev = sqrt(max(0.0, moments_history.y - moments_history.x * moments_history.x));

    float center_dev = center.a;

    // Spatial-only variance estimate (dev.rgb) has halos around edges (lighting discontinuities)

    // Temporal variance estimate with a spatial boost
    // TODO: this version reduces flicker in pica and on skeletons in battle, but has halos in cornell_box
    // dev.rgb = center_dev * dev.rgb / max(1e-8, clamp(working_luma(dev.rgb), center_dev * 0.1, center_dev * 3.0));

    // Spatiotemporal variance estimate
    // TODO: this version seems to work best, but needs to take care near sky
    // TODO: also probably needs to be rgb :P
    // dev.rgb = sqrt(dev_sum);

    // Temporal variance estimate with spatial colors
    // dev.rgb *= center_dev / max(1e-8, working_luma(dev.rgb));

    vec3 hist_dev = vec3(sqrt(abs(hist_vsum2 - hist_vsum * hist_vsum)));
    // dev.rgb *= 0.1 / max(1e-5, clamp(hist_dev, dev.rgb * 0.1, dev.rgb * 10.0));

    // float temporal_change = abs(hist_vsum - working_luma(ex.rgb)) / max(1e-8, hist_vsum + working_luma(ex.rgb));
    float temporal_change = abs(hist_vsum - working_luma(ex.rgb)) / max(1e-8, hist_vsum + working_luma(ex.rgb));
    // float temporal_change = 0.1 * abs(hist_vsum - working_luma(ex.rgb)) / max(1e-5, working_luma(dev.rgb));
    // temporal_change = 0.02 * temporal_change / max(1e-5, working_luma(dev.rgb));
    // temporal_change = WaveActiveSum(temporal_change) / WaveActiveSum(1);
#endif

    const float rt_invalid = saturate(sqrt(safeTexelFetch(rt_history_invalidity_tex, ivec2(px / 2), 0).x) * 4);
    const float current_sample_count = history.a;

    float clamp_box_size = 1 * mix(0.25, 2.0, 1.0 - rt_invalid) * mix(0.333, 1.0, saturate(reproj.w)) * 2;
    clamp_box_size = max(clamp_box_size, 0.5);

    vec4 nmin = center - dev * clamp_box_size;
    vec4 nmax = center + dev * clamp_box_size;

#if 0
    {
    	vec4 nmin2 = center;
    	vec4 nmax2 = center;

    	{const int k = 2;
        for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                vec4 neigh = linear_to_working(input_tex[px + ivec2(x, y)]);
    			nmin2 = min(nmin2, neigh);
                nmax2 = max(nmax2, neigh);
            }
        }}

        vec3 nmid = max(nmin2.rgb, nmax2.rgb, 0.5);
        nmin2.rgb = max(nmid, nmin2.rgb, 1.0);
        nmax2.rgb = max(nmid, nmax2.rgb, 1.0);

        nmin = max(nmin, nmin2);
        nmax = min(nmax, nmax2);
    }
#endif

#if 1
    vec4 clamped_history = vec4(clamp(history.rgb, nmin.rgb, nmax.rgb), history.a);
#else
    vec4 clamped_history = vec4(
        soft_color_clamp(center.rgb, history.rgb, ex.rgb, clamp_box_size * dev.rgb),
        history.a);
#endif

    /*const vec3 history_dist = abs(history.rgb - ex.rgb) / max(0.1, dev.rgb * 0.5);
    const vec3 closest_pt = clamp(history.rgb, center.rgb - dev.rgb * 0.5, center.rgb + dev.rgb * 0.5);
    clamped_history = vec4(
        max(history.rgb, closest_pt, max(0.1, 1.0, smoothstep(1.0, 3.0, history_dist))),
        history.a
    );*/

#if !USE_BBOX_CLAMP
    clamped_history = history;
#endif

    const float variance_adjusted_temporal_change = smoothstep(0.1, 1.0, 0.05 * temporal_change / center_temporal_dev);

    float max_sample_count = 32;
    max_sample_count = mix(max_sample_count, 4, variance_adjusted_temporal_change);
    // max_sample_count = mix(max_sample_count, 1, smoothstep(0.01, 0.6, 10 * temporal_change * (center_dev / max(1e-5, center_luma))));
    max_sample_count *= mix(1.0, 0.5, rt_invalid);

    // hax
    // max_sample_count = 32;

    vec3 res = mix(clamped_history.rgb, center.rgb, 1.0 / (1.0 + min(max_sample_count, current_sample_count)));
    // vec3 res = mix(clamped_history.rgb, center.rgb, 1.0 / 32);

    const float output_sample_count = min(current_sample_count, max_sample_count) + 1;
    vec4 output_ = working_to_linear(vec4(res, output_sample_count));
    safeImageStore(history_output_tex, ivec2(px), output_);

    // output = smoothstep(1.0, 3.0, history_dist);
    // output = abs(history.rgb - ex.rgb);
    // output = dev.rgb;

    // output *= reproj.z;    // debug validity
    // output *= light_stability;
    // output = smoothstep(0.0, 0.05, history_dist);
    // output = length(dev.rgb);
    // output = 1-light_stability;
    // output = control_variate_luma;
    // output = abs(rdiff);
    // output = abs(dev.rgb);
    // output = abs(hist_dev.rgb);
    // output = smoothed_dev;

    // TODO: adaptively sample according to abs(res)
    // output = abs(res);
    // output = WaveActiveSum(center.rgb) / WaveActiveSum(1);
    // output = WaveActiveSum(history.rgb) / WaveActiveSum(1);
    // output.rgb = 0.1 * temporal_change / max(1e-5, working_luma(dev.rgb));
    // output = pow(smoothstep(0.1, 1, temporal_change), 1.0);
    // output.rgb = center_temporal_dev;
    // output = center_dev / max(1e-5, center_luma);
    // output = 1 - smoothstep(0.01, 0.6, temporal_change);
    // output = pow(smoothstep(0.02, 0.6, 0.01 * temporal_change / center_temporal_dev), 0.25);
    // output = max_sample_count / 32.0;
    // output.rgb = temporal_change * 0.1;
    // output.rgb = variance_adjusted_temporal_change * 0.1;
    // output.rgb = rt_history_invalidity_tex[px / 2];
    // output.rgb = max(output.rgb, rt_invalid, 0.9);
    // output.rgb = max(output.rgb, pow(output_sample_count / 32.0, 4), 0.9);
    // output.r = 1-reproj.w;

    float temp_var = saturate(
        output_sample_count * mix(1.0, 0.5, rt_invalid) * smoothstep(0.3, 0, temporal_change) / 32.0);

    safeImageStore(output_tex, ivec2(px), vec4(output_.rgb, temp_var));

    // output_tex[px] = vec4(output.rgb, output_sample_count);
    // output_tex[px] = vec4(output.rgb, 1.0 - rt_invalid);
}

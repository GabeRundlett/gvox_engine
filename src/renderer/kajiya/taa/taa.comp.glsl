#include <renderer/kajiya/taa.inl>

#include <g_samplers>
#include "../inc/camera.glsl"
#include "../inc/color.glsl"
#include "../inc/image.glsl"
// #include "../inc/unjitter_taa.hlsl"
#include "taa_common.glsl"

DAXA_DECL_PUSH_CONSTANT(TaaComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewIndex input_image = push.uses.input_image;
daxa_ImageViewIndex reprojected_history_img = push.uses.reprojected_history_img;
daxa_ImageViewIndex reprojection_map = push.uses.reprojection_map;
daxa_ImageViewIndex closest_velocity_img = push.uses.closest_velocity_img;
daxa_ImageViewIndex velocity_history_tex = push.uses.velocity_history_tex;
daxa_ImageViewIndex depth_image = push.uses.depth_image;
daxa_ImageViewIndex smooth_var_history_tex = push.uses.smooth_var_history_tex;
daxa_ImageViewIndex input_prob_img = push.uses.input_prob_img;
daxa_ImageViewIndex temporal_output_tex = push.uses.temporal_output_tex;
daxa_ImageViewIndex this_frame_output_img = push.uses.this_frame_output_img;
daxa_ImageViewIndex smooth_var_output_tex = push.uses.smooth_var_output_tex;
daxa_ImageViewIndex temporal_velocity_output_tex = push.uses.temporal_velocity_output_tex;

// Apply at spatial kernel to the current frame, "un-jittering" it.
#define FILTER_CURRENT_FRAME 1

#define USE_ACCUMULATION 1
#define RESET_ACCUMULATION 0
#define USE_NEIGHBORHOOD_CLAMPING 1
#define TARGET_SAMPLE_COUNT 8

// If 1, outputs the input verbatim
// if N > 1, exponentially blends approximately N frames together without any clamping
#define SHORT_CIRCUIT 0

// Whether to use the input probability calculated in `input_prob.hlsl` and the subsequent filters.
// Necessary for stability of temporal super-resolution.
#define USE_CONFIDENCE_BASED_HISTORY_BLEND 1

#define INPUT_TEX input_image
#define INPUT_REMAP InputRemap

// Draw a rectangle indicating the current frame index. Useful for debugging frame drops.
#define USE_FRAME_INDEX_INDICATOR_BAR 0

vec4 InputRemap_remap(vec4 v) {
    return vec4(sRGB_to_YCbCr(decode_rgb(v.rgb)), 1);
}
vec4 fetch_history(ivec2 px) {
    return safeTexelFetch(reprojected_history_img, px, 0);
}

vec4 fetch_blurred_history(ivec2 px, int k, float sigma) {
    const vec3 center = fetch_history(px).rgb;

    vec4 csum = vec4(0.0);
    float wsum = 0;

    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            vec4 c = fetch_history(px + ivec2(x, y));
            vec2 offset = vec2(x, y) * sigma;
            float w = exp(-dot(offset, offset));
            float color_diff =
                linear_to_perceptual(sRGB_to_luminance(c.rgb)) - linear_to_perceptual(sRGB_to_luminance(center));
            csum += c * w;
            wsum += w;
        }
    }

    return csum / wsum;
}

vec4 HistoryRemap_remap(vec4 v) {
    return vec4(sRGB_to_YCbCr(v.rgb), 1);
}

struct UnjitteredSampleInfo {
    vec4 color;
    float coverage;
    vec3 ex;
    vec3 ex2;
};
struct UnjitterSettings {
    float kernel_scale;
    int kernel_half_width_pixels;
};

#define REMAP_FUNC HistoryRemap_remap
UnjitteredSampleInfo sample_image_unjitter_taa(
    daxa_ImageViewIndex img,
    ivec2 output_px,
    vec2 output_tex_size,
    vec2 sample_offset_pixels,
    UnjitterSettings settings) {
    const vec2 input_tex_size = push.input_tex_size.xy; // vec2(img.size());
    const vec2 input_resolution_scale = input_tex_size / output_tex_size;
    const ivec2 base_src_px = ivec2((output_px + 0.5) * input_resolution_scale);

    // In pixel units of the destination (upsampled)
    const vec2 dst_sample_loc = output_px + 0.5;
    const vec2 base_src_sample_loc =
        (base_src_px + 0.5 + sample_offset_pixels * vec2(1, -1)) / input_resolution_scale;

    vec4 res = vec4(0.0);
    vec3 ex = vec3(0.0);
    vec3 ex2 = vec3(0.0);
    float dev_wt_sum = 0.0;
    float wt_sum = 0.0;

    // Stretch the kernel if samples become too sparse due to drastic upsampling
    // const float kernel_distance_mult = min(1.0, 1.2 * input_resolution_scale.x);

    const float kernel_distance_mult = 1.0 * settings.kernel_scale;
    // const float kernel_distance_mult = 0.3333 / 2;

    int k = settings.kernel_half_width_pixels;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            ivec2 src_px = base_src_px + ivec2(x, y);
            vec2 src_sample_loc = base_src_sample_loc + vec2(x, y) / input_resolution_scale;

            vec4 col = REMAP_FUNC(safeTexelFetch(img, src_px, 0));
            vec2 sample_center_offset = (src_sample_loc - dst_sample_loc) * kernel_distance_mult;

            float dist2 = dot(sample_center_offset, sample_center_offset);
            float dist = sqrt(dist2);

            // float wt = all(abs(sample_center_offset) < 0.83);//dist < 0.33;
            float dev_wt = exp2(-dist2 * input_resolution_scale.x);
            // float wt = mitchell_netravali(2.5 * dist * input_resolution_scale.x);
            float wt = exp2(-10 * dist2 * input_resolution_scale.x);
            // float wt = sinc(1 * dist * input_resolution_scale.x) * smoothstep(3, 0, dist * input_resolution_scale.x);
            // float wt = lanczos(2.2 * dist * input_resolution_scale.x, 3);
            // wt = max(wt, 0.0);

            res += col * wt;
            wt_sum += wt;

            ex += col.xyz * dev_wt;
            ex2 += col.xyz * col.xyz * dev_wt;
            dev_wt_sum += dev_wt;
        }
    }

    vec2 sample_center_offset = -sample_offset_pixels / input_resolution_scale * vec2(1, -1) - (base_src_sample_loc - dst_sample_loc);

    UnjitteredSampleInfo info;
    info.color = res;
    info.coverage = wt_sum;
    info.ex = ex / dev_wt_sum;
    info.ex2 = ex2 / dev_wt_sum;
    return info;
}
#undef REMAP_FUNC

vec4 fetch_reproj(ivec2 px) {
    return safeTexelFetch(reprojection_map, px, 0);
}

layout(local_size_x = TAA_WG_SIZE_X, local_size_y = TAA_WG_SIZE_Y, local_size_z = 1) in;
void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
#if USE_FRAME_INDEX_INDICATOR_BAR
    if (px.y < 50) {
        vec4 val = 0;
        if (px.x < frame_constants.frame_index * 10 % uint(output_tex_size.x)) {
            val = 1;
        }
        temporal_output_tex[px] = val;
        output_tex[px] = val;
        return;
    }
#endif

    const vec2 input_resolution_fraction = push.input_tex_size.xy / push.output_tex_size.xy;
    const uvec2 reproj_px = uvec2((vec2(px) + 0.5) * input_resolution_fraction);
    // const uvec2 reproj_px = uvec2(px * input_resolution_fraction + 0.5);

#if SHORT_CIRCUIT
    temporal_output_tex[px] = mix(input_tex[reproj_px], vec4(encode_rgb(fetch_history(px).rgb), 1), 1.0 - 1.0 / SHORT_CIRCUIT);
    output_tex[px] = temporal_output_tex[px];
    return;
#endif

    vec2 uv = get_uv(px, vec4(push.output_tex_size.xy, 1.0 / push.output_tex_size.xy));

    vec4 history_packed = fetch_history(px);
    vec3 history = history_packed.rgb;
    float history_coverage = max(0.0, history_packed.a);

    vec4 bhistory_packed = fetch_blurred_history(px, 2, 1);
    vec3 bhistory = bhistory_packed.rgb;
    vec3 bhistory_coverage = vec3(bhistory_packed.a);

    history = sRGB_to_YCbCr(history);
    bhistory = sRGB_to_YCbCr(bhistory);

    const vec4 reproj = fetch_reproj(ivec2(reproj_px));
    const vec2 reproj_xy = safeTexelFetch(closest_velocity_img, px, 0).xy;

    UnjitterSettings unjitter_settings;
    unjitter_settings.kernel_scale = 1;
    unjitter_settings.kernel_half_width_pixels = 1;
    UnjitteredSampleInfo center_sample = sample_image_unjitter_taa(
        INPUT_TEX,
        px,
        push.output_tex_size.xy,
        vec2(deref(gpu_input).halton_jitter),
        unjitter_settings);
    unjitter_settings.kernel_scale = 0.333;
    UnjitteredSampleInfo bcenter_sample = sample_image_unjitter_taa(
        INPUT_TEX,
        px,
        push.output_tex_size.xy,
        vec2(deref(gpu_input).halton_jitter),
        unjitter_settings);

    float coverage = 1;
#if FILTER_CURRENT_FRAME
    vec3 center = center_sample.color.rgb;
    coverage = center_sample.coverage;
#else
    vec3 center = sRGB_to_YCbCr(decode_rgb(safeTexelFetch(INPUT_TEX, px, 0).rgb));
#endif

    vec3 bcenter = bcenter_sample.color.rgb / bcenter_sample.coverage;

    history = mix(history, bcenter, clamp(1.0 - history_coverage, 0.0, 1.0));
    bhistory = mix(bhistory, bcenter, clamp(1.0 - bhistory_coverage, 0.0, 1.0));

    const float input_prob = safeTexelFetch(input_prob_img, ivec2(reproj_px), 0).r;

    vec3 ex = center_sample.ex;
    vec3 ex2 = center_sample.ex2;
    const vec3 var = max(0.0.xxx, ex2 - ex * ex);

    const vec3 prev_var = vec3(textureLod(daxa_sampler2D(smooth_var_history_tex, g_sampler_lnc), uv + reproj_xy, 0).x);

    // TODO: factor-out camera-only velocity
    const vec2 vel_now = safeTexelFetch(closest_velocity_img, px, 0).xy / deref(gpu_input).delta_time;
    const vec2 vel_prev = textureLod(daxa_sampler2D(velocity_history_tex, g_sampler_llc), uv + safeTexelFetch(closest_velocity_img, px, 0).xy, 0).xy;
    const float vel_diff = length((vel_now - vel_prev) / max(vec2(1.0), abs(vel_now + vel_prev)));
    const float var_blend = clamp(0.3 + 0.7 * (1 - reproj.z) + vel_diff, 0.0, 1.0);

    vec3 smooth_var = max(var, mix(prev_var, var, var_blend));

    const float var_prob_blend = clamp(input_prob, 0.0, 1.0);
    smooth_var = mix(var, smooth_var, var_prob_blend);

    const vec3 input_dev = sqrt(var);

    vec4 this_frame_result = vec4(0.0);
#define DEBUG_SHOW(value) \
    { this_frame_result = vec4((vec3)(value), 1); }

    vec3 clamped_history;

    // Perform neighborhood clamping / disocclusion rejection
    {
        // Use a narrow color bounding box to avoid disocclusions
        float box_n_deviations = 0.8;

        if (USE_CONFIDENCE_BASED_HISTORY_BLEND != 0) {
            // Expand the box based on input confidence.
            box_n_deviations = mix(box_n_deviations, 3, input_prob);
        }

        vec3 nmin = ex - input_dev * box_n_deviations;
        vec3 nmax = ex + input_dev * box_n_deviations;

#if USE_ACCUMULATION
#if USE_NEIGHBORHOOD_CLAMPING
        vec3 clamped_bhistory = clamp(bhistory, nmin, nmax);
#else
        vec3 clamped_bhistory = bhistory;
#endif

        const float clamping_event = length(max(vec3(0.0), max(bhistory - nmax, nmin - bhistory)) / max(vec3(0.01), ex));

        vec3 outlier3 = max(vec3(0.0), (max(nmin - history, history - nmax)) / (0.1 + max(max(abs(history), abs(ex)), 1e-5)));
        vec3 boutlier3 = max(vec3(0.0), (max(nmin - bhistory, bhistory - nmax)) / (0.1 + max(max(abs(bhistory), abs(ex)), 1e-5)));

        // Temporal outliers in sharp history
        float outlier = max(outlier3.x, max(outlier3.y, outlier3.z));
        // DEBUG_SHOW(outlier);

        // Temporal outliers in blurry history
        float boutlier = max(boutlier3.x, max(boutlier3.y, boutlier3.z));
        // DEBUG_SHOW(boutlier);

        const bool history_valid = all(bvec2(uv + reproj_xy == clamp(uv + reproj_xy, vec2(0.0), vec2(1.0))));

#if 1
        if (history_valid) {
            const float non_disoccluding_outliers = max(0.0, outlier - boutlier) * 10;
            // DEBUG_SHOW(non_disoccluding_outliers);

            const vec3 unclamped_history_detail = history - clamped_bhistory;

            // Temporal luminance diff, containing history edges, and peaking when
            // clamping happens.
            const float temporal_clamping_detail = length(unclamped_history_detail.x / max(1e-3, input_dev.x)) * 0.05;
            // DEBUG_SHOW(temporal_clamping_detail);

            // Close to 1.0 when temporal clamping is relatively low. Close to 0.0 when disocclusions happen.
            const float temporal_stability = clamp(1 - temporal_clamping_detail, 0.0, 1.0);
            // DEBUG_SHOW(temporal_stability);

            const float allow_unclamped_detail = clamp(non_disoccluding_outliers, 0.0, 1.0) * temporal_stability;
            // const float allow_unclamped_detail = saturate(non_disoccluding_outliers * exp2(-length(input_tex_size.xy * reproj_xy))) * temporal_stability;
            // DEBUG_SHOW(allow_unclamped_detail);

            // Clamping happens to blurry history because input is at lower fidelity (and potentially lower resolution)
            // than history (we don't have enough data to perform good clamping of high frequencies).
            // In order to keep high-resolution detail in the output, the high-frequency content is split from
            // low-frequency (`bhistory`), and then selectively re-added. The detail needs to be attenuated
            // in order not to cause false detail (which look like excessive sharpening artifacts).
            vec3 history_detail = history - bhistory;

            // Selectively stabilize some detail, allowing unclamped history
            history_detail = mix(history_detail, unclamped_history_detail, allow_unclamped_detail);

            // 0..1 value of how much clamping initially happened in the blurry history
            const float initial_bclamp_amount = clamp(dot(
                                                          clamped_bhistory - bhistory, bcenter - bhistory) /
                                                          max(1e-5, length(clamped_bhistory - bhistory) * length(bcenter - bhistory)),
                                                      0.0, 1.0);

            // Ditto, after adjusting for `allow_unclamped_detail`
            const float effective_clamp_amount = clamp(initial_bclamp_amount, 0.0, 1.0) * (1 - allow_unclamped_detail);
            // DEBUG_SHOW(effective_clamp_amount);

            // Where clamping happened to the blurry history, also remove the detail (history-bhistory)
            const float keep_detail = 1 - effective_clamp_amount;
            history_detail *= keep_detail;

            // Finally, construct the full-frequency output.
            clamped_history = clamped_bhistory + history_detail;

#if 1
            // TODO: figure out how not to over-do this with temporal super-resolution
            if (input_resolution_fraction.x < 1.0) {
                // When temporally upsampling, after a clamping event, there's pixellation
                // because we haven't accumulated enough samples yet from
                // the reduced-resolution input. Dampening history coverage when
                // clamping happens allows us to boost this convergence.

                history_coverage *= mix(
                    mix(0.0, 0.9, keep_detail), 1.0, clamp(10 * clamping_event, 0.0, 1.0));
            }
#endif
        } else {
            clamped_history = clamped_bhistory;
            coverage = 1;
            center = bcenter;
            history_coverage = 0;
        }
#else
        clamped_history = clamp(history, nmin, nmax);
#endif

        if (USE_CONFIDENCE_BASED_HISTORY_BLEND != 0) {
            // If input confidence is high, blend in unclamped history.
            clamped_history = mix(
                clamped_history,
                history,
                smoothstep(0.5, 1.0, input_prob));
        }
    }

#if RESET_ACCUMULATION
    history_coverage = 0;
#endif

    float total_coverage = max(1e-5, history_coverage + coverage);
    vec3 temporal_result = (clamped_history * history_coverage + center) / total_coverage;

    const float max_coverage = max(2, TARGET_SAMPLE_COUNT / (input_resolution_fraction.x * input_resolution_fraction.y));

    total_coverage = min(max_coverage, total_coverage);

    coverage = total_coverage;
#else
        vec3 temporal_result = center / coverage;
#endif
    safeImageStore(smooth_var_output_tex, px, vec4(smooth_var, 0.0));

    temporal_result = YCbCr_to_sRGB(temporal_result);
    temporal_result = encode_rgb(temporal_result);
    temporal_result = max(vec3(0.0), temporal_result);

    this_frame_result.rgb = mix(temporal_result, this_frame_result.rgb, this_frame_result.a);

    safeImageStore(temporal_output_tex, px, vec4(temporal_result, coverage));
    safeImageStore(this_frame_output_img, px, this_frame_result);

    vec2 vel_out = reproj_xy;
    float vel_out_depth = 0;

    // It's critical that this uses the closest depth since it's compared to closest depth
    safeImageStore(temporal_velocity_output_tex, px, vec4(safeTexelFetch(closest_velocity_img, px, 0).xy / deref(gpu_input).delta_time, 0.0, 0.0));
}

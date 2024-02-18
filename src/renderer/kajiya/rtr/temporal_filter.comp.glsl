#include <renderer/kajiya/rtr.inl>

#include <renderer/kajiya/inc/camera.glsl>
#include <g_samplers>
// #include <utilities/gpu/uv.glsl>
// #include "../inc/frame_constants.glsl"
#include "../inc/color.glsl"
#include "../inc/bilinear.glsl"
// #include <utilities/gpu/soft_color_clamp.glsl>
#include "../inc/image.glsl"
#include "../inc/brdf.glsl"
#include "../inc/gbuffer.glsl"
#include "rtr_settings.glsl"

#include "../inc/working_color_space.glsl"
#include "../inc/safety.glsl"

// Use this after tweaking all the spec.
#define linear_to_working linear_rgb_to_crunched_luma_chroma
#define working_to_linear crunched_luma_chroma_to_linear_rgb

// #define linear_to_working linear_rgb_to_linear_rgb
// #define working_to_linear linear_rgb_to_linear_rgb

#define USE_DUAL_REPROJECTION 1
#define USE_NEIGHBORHOOD_CLAMP 1

DAXA_DECL_PUSH_CONSTANT(RtrTemporalFilterComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewIndex input_tex = push.uses.input_tex;
daxa_ImageViewIndex history_tex = push.uses.history_tex;
daxa_ImageViewIndex depth_tex = push.uses.depth_tex;
daxa_ImageViewIndex ray_len_tex = push.uses.ray_len_tex;
daxa_ImageViewIndex reprojection_tex = push.uses.reprojection_tex;
daxa_ImageViewIndex refl_restir_invalidity_tex = push.uses.refl_restir_invalidity_tex;
daxa_ImageViewIndex gbuffer_tex = push.uses.gbuffer_tex;
daxa_ImageViewIndex output_tex = push.uses.output_tex;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    const uvec2 px = gl_GlobalInvocationID.xy;
#if !RTR_USE_TEMPORAL_FILTERS
    output_tex[px] = vec4(input_tex[px].rgb, 128);
    return;
#endif

    const vec4 center = linear_to_working(safeTexelFetch(input_tex, ivec2(px), 0));

    float refl_ray_length = clamp(safeTexelFetch(ray_len_tex, ivec2(px), 0).x, 0, 1e3);

    // TODO: run a small edge-aware soft-min filter of ray length.
    // The `WaveActiveMin` below improves flat rough surfaces, but is not correct across discontinuities.
    // refl_ray_length = WaveActiveMin(refl_ray_length);

    vec2 uv = get_uv(px, push.output_tex_size);

    const float center_depth = safeTexelFetch(depth_tex, ivec2(px), 0).r;
    const ViewRayContext view_ray_context = vrc_from_uv_and_depth(gpu_input, uv, center_depth);
    const vec3 reflector_vs = ray_hit_vs(view_ray_context);
    const vec3 reflection_hit_vs = reflector_vs + ray_dir_vs(view_ray_context) * refl_ray_length;

    const vec4 reflection_hit_cs = deref(gpu_input).player.cam.view_to_sample * vec4(reflection_hit_vs, 1);
    const vec4 prev_hit_cs = deref(gpu_input).player.cam.clip_to_prev_clip * reflection_hit_cs;
    vec2 hit_prev_uv = cs_to_uv(prev_hit_cs.xy / prev_hit_cs.w);

    const vec4 prev_reflector_cs = deref(gpu_input).player.cam.clip_to_prev_clip * view_ray_context.ray_hit_cs;
    const vec2 reflector_prev_uv = cs_to_uv(prev_reflector_cs.xy / prev_reflector_cs.w);

    vec4 reproj = safeTexelFetch(reprojection_tex, ivec2(px), 0);

    const vec2 reflector_move_rate = vec2(min(1.0, length(reproj.xy) / length(reflector_prev_uv - uv)));
    hit_prev_uv = mix(uv, hit_prev_uv, reflector_move_rate);

    const uint quad_reproj_valid_packed = uint(reproj.z * 15.0 + 0.5);
    const vec4 quad_reproj_valid = vec4(notEqual(quad_reproj_valid_packed & uvec4(1, 2, 4, 8), uvec4(0)));

    const vec4 history_mult = vec4((deref(gpu_input).pre_exposure_delta).xxx, 1);

    vec4 history0 = vec4(0.0);
    float history0_valid = 1;
#if 0
        history0 = linear_to_working(history_tex.SampleLevel(sampler_lnc, uv + reproj.xy, 0) * history_mult);
#else
    if (0 == quad_reproj_valid_packed) {
        // Everything invalid
        history0_valid = 0;
    } else if (15 == quad_reproj_valid_packed) {
        // history0 = history_tex.SampleLevel(sampler_lnc, uv + reproj.xy, 0) * history_mult;
        history0 = max(vec4(0.0), image_sample_catmull_rom_5tap(IdentityImageRemap_remap)(
                                history_tex,
                                g_sampler_lnc,
                                uv + reproj.xy,
                                push.output_tex_size.xy)) *
                   history_mult;
    } else {
        vec4 quad_reproj_valid = vec4(notEqual(quad_reproj_valid_packed & uvec4(1, 2, 4, 8), uvec4(0)));

        const Bilinear bilinear = get_bilinear_filter(uv + reproj.xy, push.output_tex_size.xy);
        vec4 s00 = safeTexelFetch(history_tex, ivec2(bilinear.origin) + ivec2(0, 0), 0) * history_mult;
        vec4 s10 = safeTexelFetch(history_tex, ivec2(bilinear.origin) + ivec2(1, 0), 0) * history_mult;
        vec4 s01 = safeTexelFetch(history_tex, ivec2(bilinear.origin) + ivec2(0, 1), 0) * history_mult;
        vec4 s11 = safeTexelFetch(history_tex, ivec2(bilinear.origin) + ivec2(1, 1), 0) * history_mult;
        vec4 weights = get_bilinear_custom_weights(bilinear, quad_reproj_valid);

        if (dot(weights, vec4(1.0)) > 1e-5) {
            history0 = apply_bilinear_custom_weights(s00, s10, s01, s11, weights, true);
        } else {
            // Invalid, but we have to return something.
            history0 = (s00 + s10 + s01 + s11) / 4;
        }
    }
    history0 = linear_to_working(history0);
#endif

    vec4 history1 = linear_to_working(textureLod(daxa_sampler2D(history_tex, g_sampler_lnc), hit_prev_uv, 0) * history_mult);
    /*vec4 history1 = linear_to_working(max(0.0, image_sample_catmull_rom_5tap(
        history_tex,
        sampler_lnc,
        hit_prev_uv,
        push.output_tex_size.xy,
        IdentityImageRemap::create()
    )) * history_mult);*/

    float history1_valid = float(quad_reproj_valid_packed == 15);

    vec4 history0_reproj = textureLod(daxa_sampler2D(reprojection_tex, g_sampler_lnc), uv + reproj.xy, 0);
    vec4 history1_reproj = textureLod(daxa_sampler2D(reprojection_tex, g_sampler_lnc), hit_prev_uv, 0);

    vec4 vsum = 0.0.xxxx;
    vec4 vsum2 = 0.0.xxxx;
    float wsum = 0.0;

    const int k = 1;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            const ivec2 sample_px = ivec2(px) + ivec2(x, y) * 1;
            const float sample_depth = safeTexelFetch(depth_tex, sample_px, 0).r;

            vec4 neigh = linear_to_working(safeTexelFetch(input_tex, sample_px, 0));
            float w = 1; // exp(-3.0 * float(x * x + y * y) / float((k+1.) * (k+1.)));

            w *= exp2(-200.0 * abs(/*center_normal_vs.z **/ (center_depth / sample_depth - 1.0)));

            vsum += neigh * w;
            vsum2 += neigh * neigh * w;
            wsum += w;
        }
    }

    vec4 ex = vsum / wsum;
    vec4 ex2 = vsum2 / wsum;
    vec4 dev = sqrt(max(0.0.xxxx, ex2 - ex * ex));

    GbufferData gbuffer = unpack(GbufferDataPacked(safeTexelFetchU(gbuffer_tex, ivec2(px), 0)));

    const float restir_invalidity = safeTexelFetch(refl_restir_invalidity_tex, ivec2(px / 2), 0).x;

    float box_size = 1;
    // float n_deviations = 1;
    float n_deviations = mix(select(reproj.z > 0, 2, 1.25), 0.625, restir_invalidity);

    float wo_similarity;
    {
        // TODO: take object motion into account too
        const vec3 current_wo = normalize(ray_hit_ws(view_ray_context) - get_eye_position(gpu_input));
        const vec3 prev_wo = normalize(ray_hit_ws(view_ray_context) - get_prev_eye_position(gpu_input));

        const float clamped_roughness = max(0.1, gbuffer.roughness);

        wo_similarity =
            pow(saturate(SpecularBrdf_ggx_ndf_0_1(clamped_roughness * clamped_roughness, dot(current_wo, prev_wo))), 32);
    }

    float h0diff = length((history0.xyz - ex.xyz) / dev.xyz);
    float h1diff = length((history1.xyz - ex.xyz) / dev.xyz);

#if USE_DUAL_REPROJECTION
    float h0_score =
        1.0
        // Favor direct reprojection at high roughness.
        * smoothstep(0, 0.5, sqrt(gbuffer.roughness))
        //* sqrt(gbuffer.roughness)
        // Except when under a lot of parallax motion.
        * mix(wo_similarity, 1, sqrt(gbuffer.roughness));
    float h1_score =
        (1 - h0_score) * mix(
                             1,
                             // Don't use the parallax-based reprojection when direct reprojection has
                             // much lower difference to the new frame's mean.
                             smoothstep(0, 1, h0diff - h1diff),
                             // ... except at low roughness values, where we really want to use
                             // the parallax-based reprojection.
                             smoothstep(0.0, 0.15, sqrt(gbuffer.roughness)));
#else
    float h0_score = 1;
    float h1_score = 0;
#endif

    h0_score *= history0_valid;
    h1_score *= history1_valid;

    const float score_sum = h0_score + h1_score;
    h0_score /= score_sum;
    h1_score = 1 - h0_score;

    if (!(h0_score < 1.001)) {
        h0_score = 1;
        h1_score = 0;
    }

    vec4 clamped_history0 = history0;
    vec4 clamped_history1 = history1;

#if 0
	vec4 nmin = center - dev * box_size * n_deviations;
	vec4 nmax = center + dev * box_size * n_deviations;

    clamped_history0.rgb = clamp(history0.rgb, nmin.rgb, nmax.rgb);
    clamped_history1.rgb = clamp(history1.rgb, nmin.rgb, nmax.rgb);
#else
    clamped_history0.rgb = soft_color_clamp(center.rgb, history0.rgb, ex.rgb, dev.rgb * n_deviations);
    clamped_history1.rgb = soft_color_clamp(center.rgb, history1.rgb, ex.rgb, dev.rgb * n_deviations);
#endif

    vec4 unclamped_history = history0 * h0_score + history1 * h1_score;
    vec4 clamped_history = clamped_history0 * h0_score + clamped_history1 * h1_score;

#if !USE_NEIGHBORHOOD_CLAMP
    clamped_history = history0 * h0_score + history1 * h1_score;
#endif

    float max_sample_count = 16;
    const float current_sample_count =
        clamped_history.a * saturate(h0_score * history0_valid + h1_score * history1_valid);

    vec4 filtered_center = center;
    vec4 res = mix(
        clamped_history,
        filtered_center,
        1.0 / (1.0 + min(max_sample_count, current_sample_count * mix(wo_similarity, 1, 0.5))));
    res.w = min(current_sample_count, max_sample_count) + 1;
    // res.w = sample_count + 1;
    // res.w = refl_ray_length * 20;
    // res.w = (dev / ex).x * 16;
    // res.w = 2 * exp2(-20 * (dev / ex).x);
    // res.w = refl_ray_length;

    // res.rgb = working_to_linear(dev).rgb / max(1e-8, working_to_linear(ex).rgb);
    res = working_to_linear(res);

    safeImageStore(output_tex, ivec2(px), max(0.0.xxxx, res));
}

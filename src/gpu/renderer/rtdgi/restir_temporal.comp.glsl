#include <shared/app.inl>

#include <utils/math.glsl>
// #include <utils/uv.glsl>
// #include <utils/pack_unpack.glsl>
// #include <utils/frame_constants.glsl>
#include <utils/gbuffer.glsl>
#include <utils/brdf.glsl>
#include <utils/brdf_lut.glsl>
#include <utils/layered_brdf.glsl>
// #include <utils/blue_noise.glsl>
// #include <utils/atmosphere.glsl>
// #include <utils/sun.glsl>
// #include <utils/lights/triangle.glsl>
#include <utils/reservoir.glsl>
// #include "../ircache/bindings.hlsl"
#include "near_field_settings.glsl"
#include "rtdgi_restir_settings.glsl"
#include "rtdgi_common.glsl"

#include <utils/safety.glsl>
#include <utils/downscale.glsl>
#include <utils/rt.glsl>

const float SKY_DIST = 1e4;

uvec2 reservoir_payload_to_px(uint payload) {
    return uvec2(payload & 0xffff, payload >> 16);
}

struct TraceResult {
    vec3 out_value;
    vec3 hit_normal_ws;
    float inv_pdf;
    // bool prev_sample_valid;
};

TraceResult do_the_thing(uvec2 px, inout uint rng, RayDesc outgoing_ray, vec3 primary_hit_normal) {
    const vec4 candidate_radiance_inv_pdf = safeTexelFetch(candidate_radiance_tex, ivec2(px), 0);
    TraceResult result;
    result.out_value = candidate_radiance_inv_pdf.rgb;
    result.inv_pdf = 1;
    result.hit_normal_ws = direction_view_to_world(globals, safeTexelFetch(candidate_normal_tex, ivec2(px), 0).xyz);
    return result;
}

ivec2 get_rpx_offset(uint sample_i, uint frame_index) {
    const ivec2 offsets[4] = {
        ivec2(-1, -1),
        ivec2(1, 1),
        ivec2(-1, 1),
        ivec2(1, -1),
    };

    const ivec2 reservoir_px_offset_base =
        offsets[frame_index & 3] + offsets[(sample_i + (frame_index ^ 1)) & 3];

    return select(bvec2(sample_i == 0), ivec2(0), ivec2(reservoir_px_offset_base));
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    uvec2 px = gl_GlobalInvocationID.xy;
    const ivec2 hi_px_offset = ivec2(HALFRES_SUBSAMPLE_OFFSET);
    const uvec2 hi_px = px * 2 + hi_px_offset;

    float depth = safeTexelFetch(depth_tex, ivec2(hi_px), 0).r;

    if (0.0 == depth) {
        safeImageStore(radiance_out_tex, ivec2(px), vec4(0.0.xxx, -SKY_DIST));
        safeImageStore(hit_normal_output_tex, ivec2(px), 0.0.xxxx);
        safeImageStoreU(reservoir_out_tex, ivec2(px), uvec4(0));
        return;
    }

    const vec2 uv = get_uv(hi_px, push.gbuffer_tex_size);
    const ViewRayContext view_ray_context = vrc_from_uv_and_biased_depth(globals, uv, depth);
    const vec3 normal_vs = safeTexelFetch(half_view_normal_tex, ivec2(px), 0).xyz;
    const vec3 normal_ws = direction_view_to_world(globals, normal_vs);
    const daxa_f32mat3x3 tangent_to_world = tbn_from_normal(normal_ws);
    const vec3 refl_ray_origin_ws = biased_secondary_ray_origin_ws_with_normal(view_ray_context, normal_ws);

    const vec3 hit_offset_ws = safeTexelFetch(candidate_hit_tex, ivec2(px), 0).xyz;
    vec3 outgoing_dir = normalize(hit_offset_ws);

    uint rng = hash3(uvec3(px, deref(gpu_input).frame_index));

    uvec2 src_px_sel = px;
    vec3 radiance_sel = vec3(0);
    vec3 ray_orig_sel_ws = vec3(0);
    vec3 ray_hit_sel_ws = vec3(1);
    vec3 hit_normal_sel = vec3(1);
    // bool prev_sample_valid = false;

    Reservoir1sppStreamState stream_state = Reservoir1sppStreamState_create();
    Reservoir1spp reservoir = Reservoir1spp_create();
    const uint reservoir_payload = px.x | (px.y << 16);

    if (is_rtdgi_tracing_frame()) {
        RayDesc outgoing_ray;
        outgoing_ray.Direction = outgoing_dir;
        outgoing_ray.Origin = refl_ray_origin_ws;
        outgoing_ray.TMin = 0;
        outgoing_ray.TMax = SKY_DIST;

        const float hit_t = length(hit_offset_ws);

        TraceResult result = do_the_thing(px, rng, outgoing_ray, normal_ws);

        /*if (USE_SPLIT_RT_NEAR_FIELD) {
            const float NEAR_FIELD_FADE_OUT_END = -view_ray_context.ray_hit_vs().z * (SSGI_NEAR_FIELD_RADIUS * gbuffer_tex_size.w * 0.5);
            const float NEAR_FIELD_FADE_OUT_START = NEAR_FIELD_FADE_OUT_END * 0.5;
            float infl = hit_t / (SSGI_NEAR_FIELD_RADIUS * gbuffer_tex_size.w * 0.5) / -view_ray_context.ray_hit_vs().z;
            result.out_value *= smoothstep(0.0, 1.0, infl);
        }*/

        const float p_q = 1.0 * max(0, sRGB_to_luminance(result.out_value))
                          // Note: using max(0, dot) reduces noise in easy areas,
                          // but then increases it in corners by undersampling grazing angles.
                          // Effectively over time the sampling turns cosine-distributed, which
                          // we avoided doing in the first place.
                          * step(0, dot(outgoing_dir, normal_ws));

        const float inv_pdf_q = result.inv_pdf;

        radiance_sel = result.out_value;
        ray_orig_sel_ws = outgoing_ray.Origin;
        ray_hit_sel_ws = outgoing_ray.Origin + outgoing_ray.Direction * hit_t;
        hit_normal_sel = result.hit_normal_ws;
        // prev_sample_valid = result.prev_sample_valid;

        init_with_stream(reservoir, p_q, inv_pdf_q, stream_state, reservoir_payload);

        float rl = mix(safeTexelFetch(candidate_history_tex, ivec2(px), 0).y, sqrt(hit_t), 0.05);
        safeImageStore(candidate_out_tex, ivec2(px), vec4(sqrt(hit_t), rl, 0, 0));
    }

    const float rt_invalidity = sqrt(saturate(safeTexelFetch(rt_invalidity_tex, ivec2(px), 0).y));

    const bool use_resampling = DIFFUSE_GI_USE_RESTIR != 0;
    // const bool use_resampling = false;

    // 1 (center) plus offset samples
    const uint MAX_RESOLVE_SAMPLE_COUNT =
        select(RESTIR_TEMPORAL_USE_PERMUTATIONS, 5, 1);

    float center_M = 0;

    if (use_resampling) {
        for (
            uint sample_i = 0;
            sample_i < MAX_RESOLVE_SAMPLE_COUNT
            // Use permutation sampling, but only up to a certain M; those are lower quality,
            // so we want to be rather conservative.
            && stream_state.M_sum < 1.25 * RESTIR_TEMPORAL_M_CLAMP;
            ++sample_i) {
            const ivec2 rpx_offset = get_rpx_offset(sample_i, deref(gpu_input).frame_index);
            if (sample_i > 0 && all(equal(rpx_offset, ivec2(0)))) {
                // No point using the center sample twice
                continue;
            }

            const vec4 reproj = safeTexelFetch(reprojection_tex, ivec2(hi_px + rpx_offset * 2), 0);

            // Can't use linear interpolation, but we can interpolate stochastically instead
            // const vec2 reproj_rand_offset = vec2(uint_to_u01_float(hash1_mut(rng)), uint_to_u01_float(hash1_mut(rng))) - 0.5;
            // Or not at all.
            const vec2 reproj_rand_offset = vec2(0.0);

            const uvec2 xor_seq[4] = {
                uvec2(3, 3),
                uvec2(2, 1),
                uvec2(1, 2),
                uvec2(3, 3),
            };
            const uvec2 permutation_xor_val =
                xor_seq[deref(gpu_input).frame_index & 3];

            const ivec2 permuted_reproj_px = ivec2(floor(
                select(bvec2(sample_i == 0), px
                       // My poor approximation of permutation sampling.
                       // https://twitter.com/more_fps/status/1457749362025459715
                       //
                       // When applied everywhere, it does nicely reduce noise, but also makes the GI less reactive
                       // since we're effectively increasing the lifetime of the most attractive samples.
                       // Where it does come in handy though is for boosting convergence rate for newly revealed
                       // locations.
                       ,
                       ((px + rpx_offset) ^ permutation_xor_val)) +
                push.gbuffer_tex_size.xy * reproj.xy * 0.5 + reproj_rand_offset + 0.5));

            const ivec2 rpx = permuted_reproj_px + rpx_offset;
            const uvec2 rpx_hi = rpx * 2 + hi_px_offset;

            const ivec2 permuted_neighbor_px = ivec2(floor(
                select(bvec2(sample_i == 0), px
                       // ditto
                       ,
                       ((px + rpx_offset) ^ permutation_xor_val)) +
                0.5));

            const ivec2 neighbor_px = permuted_neighbor_px + rpx_offset;
            const uvec2 neighbor_px_hi = neighbor_px * 2 + hi_px_offset;

            // WRONG. needs previous normal
            // const vec3 sample_normal_vs = half_view_normal_tex[rpx];
            // // Note: also doing this for sample 0, as under extreme aliasing,
            // // we can easily get bad samples in.
            // if (dot(sample_normal_vs, normal_vs) < 0.7) {
            //     continue;
            // }

            Reservoir1spp r = Reservoir1spp_from_raw(safeTexelFetchU(reservoir_history_tex, ivec2(rpx), 0).xy);
            const uvec2 spx = reservoir_payload_to_px(r.payload);

            float visibility = 1;
            // float relevance = select(sample_i == 0, 1, 0.5);
            float relevance = 1;

            // const vec2 sample_uv = get_uv(rpx_hi, push.gbuffer_tex_size);
            const float sample_depth = safeTexelFetch(depth_tex, ivec2(neighbor_px_hi), 0).r;

            // WRONG: needs previous depth
            // if (length(prev_ray_orig_and_dist.xyz - refl_ray_origin_ws) > 0.1 * -view_ray_context.ray_hit_vs().z) {
            //     // Reject disocclusions
            //     continue;
            // }

            const vec3 prev_ray_orig = safeTexelFetch(ray_orig_history_tex, ivec2(spx), 0).xyz;
            if (length(prev_ray_orig - refl_ray_origin_ws) > 0.1 * -ray_hit_vs(view_ray_context).z) {
                // Reject disocclusions
                continue;
            }

            // Note: also doing this for sample 0, as under extreme aliasing,
            // we can easily get bad samples in.
            if (0 == sample_depth) {
                continue;
            }

            // TODO: some more rejection based on the reprojection map.
            // This one is not enough ("battle", buttom of tower).
            if (reproj.z == 0) {
                continue;
            }

#if 1
            relevance *= 1 - smoothstep(0.0, 0.1, inverse_depth_relative_diff(depth, sample_depth));
#else
            if (inverse_depth_relative_diff(depth, sample_depth) > 0.2) {
                continue;
            }
#endif

            const vec3 sample_normal_vs = safeTexelFetch(half_view_normal_tex, ivec2(neighbor_px), 0).rgb;
            const float normal_similarity_dot = max(0.0, dot(sample_normal_vs, normal_vs));

// Increases noise, but prevents leaking in areas of geometric complexity
#if 1
            // High cutoff seems unnecessary. Had it at 0.9 before.
            const float normal_cutoff = 0.2;
            if (sample_i != 0 && normal_similarity_dot < normal_cutoff) {
                continue;
            }
#endif

            relevance *= pow(normal_similarity_dot, 4);

            // TODO: this needs fixing with reprojection
            // const ViewRayContext sample_ray_ctx = ViewRayContext::from_uv_and_depth(sample_uv, sample_depth);

            const vec4 sample_hit_ws_and_dist = safeTexelFetch(ray_history_tex, ivec2(spx), 0) + vec4(prev_ray_orig, 0.0);
            const vec3 sample_hit_ws = sample_hit_ws_and_dist.xyz;
            // const vec3 prev_dir_to_sample_hit_unnorm_ws = sample_hit_ws - sample_ray_ctx.ray_hit_ws();
            // const vec3 prev_dir_to_sample_hit_ws = normalize(prev_dir_to_sample_hit_unnorm_ws);
            const float prev_dist = sample_hit_ws_and_dist.w;

            // Note: `hit_normal_history_tex` is not reprojected.
            const vec4 sample_hit_normal_ws_dot = decode_hit_normal_and_dot(safeTexelFetch(hit_normal_history_tex, ivec2(spx), 0));

            /*if (sample_i > 0 && !(prev_dist > 1e-4)) {
                continue;
            }*/

            const vec3 dir_to_sample_hit_unnorm = sample_hit_ws - refl_ray_origin_ws;
            const float dist_to_sample_hit = length(dir_to_sample_hit_unnorm);
            const vec3 dir_to_sample_hit = normalize(dir_to_sample_hit_unnorm);

            const float center_to_hit_vis = -dot(sample_hit_normal_ws_dot.xyz, dir_to_sample_hit);
            // const float prev_to_hit_vis = -dot(sample_hit_normal_ws_dot.xyz, prev_dir_to_sample_hit_ws);

            const vec4 prev_rad =
                safeTexelFetch(radiance_history_tex, ivec2(spx), 0) * vec4((deref(gpu_input).pre_exposure_delta).xxx, 1);

            // From the ReSTIR paper:
            // With temporal reuse, the number of candidates M contributing to the
            // pixel can in theory grow unbounded, as each frame always combines
            // its reservoir with the previous frame’s. This causes (potentially stale)
            // temporal samples to be weighted disproportionately high during
            // resampling. To fix this, we simply clamp the previous frame’s M
            // to at most 20× of the current frame’s reservoir’s M

            r.M = max(0, min(r.M, exp2(log2(RESTIR_TEMPORAL_M_CLAMP) * (1.0 - rt_invalidity))));
            // r.M = min(r.M, RESTIR_TEMPORAL_M_CLAMP);
            // r.M = min(r.M, 0.1);

            const float p_q = 1 * max(0, sRGB_to_luminance(prev_rad.rgb))
                              // Note: using max(0, dot) reduces noise in easy areas,
                              // but then increases it in corners by undersampling grazing angles.
                              // Effectively over time the sampling turns cosine-distributed, which
                              // we avoided doing in the first place.
                              * step(0, dot(dir_to_sample_hit, normal_ws));

            float jacobian = 1;

            // Note: needed for sample 0 too due to temporal jitter.
            {
                // Distance falloff. Needed to avoid leaks.
                jacobian *= clamp(prev_dist / dist_to_sample_hit, 1e-4, 1e4);
                jacobian *= jacobian;

                // N of hit dot -L. Needed to avoid leaks. Without it, light "hugs" corners.
                //
                jacobian *= clamp(center_to_hit_vis / sample_hit_normal_ws_dot.w, 0, 1e4);
            }

            // Fixes boiling artifacts near edges. Unstable jacobians,
            // but also effectively reduces reliance on reservoir exchange
            // in tight corners, which is desirable since the well-distributed
            // raw samples thrown at temporal filters will do better.
            if (RTDGI_RESTIR_USE_JACOBIAN_BASED_REJECTION) {
                // Clamp neighbors give us a hit point that's considerably easier to sample
                // from our own position than from the neighbor. This can cause some darkening,
                // but prevents fireflies.
                //
                // The darkening occurs in corners, where micro-bounce should be happening instead.

#if 1
                // Doesn't over-darken corners as much
                jacobian = min(jacobian, RTDGI_RESTIR_JACOBIAN_BASED_REJECTION_VALUE);
#else
                // Slightly less noise
                if (jacobian > RTDGI_RESTIR_JACOBIAN_BASED_REJECTION_VALUE) {
                    continue;
                }
#endif
            }

            r.M *= relevance;

            if (0 == sample_i) {
                center_M = r.M;
            }

            // TODO: Figure out how there could be hits closer than 1 voxel even on flat surfaces.
            safeImageStore(rtdgi_debug_image, ivec2(px), vec4(vec3(dist_to_sample_hit < 1.0 / VOXEL_SCL), 1));

            if (update_with_stream(reservoir,
                                   r, p_q, jacobian * visibility,
                                   stream_state, reservoir_payload, rng)) {
                outgoing_dir = dir_to_sample_hit;
                src_px_sel = rpx;
                radiance_sel = prev_rad.rgb;
                ray_orig_sel_ws = prev_ray_orig;
                ray_hit_sel_ws = sample_hit_ws;
                hit_normal_sel = sample_hit_normal_ws_dot.xyz;
            }
        }

        finish_stream(reservoir, stream_state);
        reservoir.W = min(reservoir.W, RESTIR_RESERVOIR_W_CLAMP);
    }

    // TODO: this results in M being accumulated at a slower rate, although finally reaching
    // the limit we're after. What it does is practice is slow down the kernel tightening
    // in the subsequent spatial reservoir resampling.
    reservoir.M = center_M + 0.5;
    // reservoir.M = center_M + 1;

    RayDesc outgoing_ray;
    outgoing_ray.Direction = outgoing_dir;
    outgoing_ray.Origin = refl_ray_origin_ws;
    outgoing_ray.TMin = 0;

    const vec4 hit_normal_ws_dot = vec4(hit_normal_sel, -dot(hit_normal_sel, outgoing_ray.Direction));

    safeImageStore(radiance_out_tex, ivec2(px), vec4(radiance_sel, dot(normal_ws, outgoing_ray.Direction)));
    safeImageStore(ray_orig_output_tex, ivec2(px), vec4(ray_orig_sel_ws, 0.0));
    safeImageStore(hit_normal_output_tex, ivec2(px), encode_hit_normal_and_dot(hit_normal_ws_dot));
    safeImageStore(ray_output_tex, ivec2(px), vec4(ray_hit_sel_ws - ray_orig_sel_ws, length(ray_hit_sel_ws - refl_ray_origin_ws)));
    safeImageStoreU(reservoir_out_tex, ivec2(px), uvec4(as_raw(reservoir), 0, 0));

    TemporalReservoirOutput res_packed;
    res_packed.depth = depth;
    res_packed.ray_hit_offset_ws = ray_hit_sel_ws - ray_hit_ws(view_ray_context);
    res_packed.luminance = max(0.0, sRGB_to_luminance(radiance_sel));
    res_packed.hit_normal_ws = hit_normal_ws_dot.xyz;
    safeImageStoreU(temporal_reservoir_packed_tex, ivec2(px), as_raw(res_packed));
}

#include <shared/app.inl>

#include <utils/math.glsl>
#include <utils/color.glsl>
// #include <utils/samplers.glsl>
// #include <utils/frame_constants.glsl>
// #include <utils/pack_unpack.glsl>
#include <utils/brdf.glsl>
#include <utils/brdf_lut.glsl>
#include <utils/layered_brdf.glsl>
// #include <utils/uv.glsl>
// #include <utils/hash.glsl>
#include <utils/reservoir.glsl>
#include "rtdgi_restir_settings.glsl"
#include "rtdgi_common.glsl"
#include <utils/occlusion_raymarch.glsl>

#include <utils/safety.glsl>
#include <utils/downscale.glsl>
#include <utils/rt.glsl>

#define USE_SSAO_WEIGHING 1
#define ALLOW_REUSE_OF_BACKFACING 1

uvec2 reservoir_payload_to_px(uint payload) {
    return uvec2(payload & 0xffff, payload >> 16);
}

// Two-thirds of SmeLU
float normal_inluence_nonlinearity(float x, float b) {
    return select(x < -b, 0, (x + b) * (x + b) / (4 * b));
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    uvec2 px = gl_GlobalInvocationID.xy;
    const uvec2 hi_px = px * 2 + HALFRES_SUBSAMPLE_OFFSET;

    float depth = safeTexelFetch(depth_tex, ivec2(hi_px), 0).r;

    const uint seed = deref(gpu_input).frame_index + push.spatial_reuse_pass_idx * 123;
    uint rng = hash3(uvec3(px, seed));

    vec2 uv = get_uv(hi_px, push.gbuffer_tex_size);
    // uv = uv_to_ss(gpu_input, uv, push.gbuffer_tex_size);
    const ViewRayContext view_ray_context = vrc_from_uv_and_depth(globals, uv_to_ss(gpu_input, uv, push.gbuffer_tex_size), depth);

    const vec3 center_normal_vs = safeTexelFetch(half_view_normal_tex, ivec2(px), 0).rgb;
    const vec3 center_normal_ws = direction_view_to_world(globals, center_normal_vs);
    const float center_depth = safeTexelFetch(half_depth_tex, ivec2(px), 0).r;
    const float center_ssao = safeTexelFetch(half_ssao_tex, ivec2(px), 0).r;

    Reservoir1sppStreamState stream_state = Reservoir1sppStreamState_create();
    Reservoir1spp reservoir = Reservoir1spp_create();

    vec3 dir_sel = vec3(1);

    float sample_radius_offset = uint_to_u01_float(hash1_mut(rng));

    Reservoir1spp center_r = Reservoir1spp_from_raw(safeTexelFetchU(reservoir_input_tex, ivec2(px), 0).xy);

    float kernel_tightness = 1.0 - center_ssao;

    const uint SAMPLE_COUNT_PASS0 = 8;
    const uint SAMPLE_COUNT_PASS1 = 5;

    const float MAX_INPUT_M_IN_PASS0 = RESTIR_TEMPORAL_M_CLAMP;
    const float MAX_INPUT_M_IN_PASS1 = MAX_INPUT_M_IN_PASS0 * SAMPLE_COUNT_PASS0;
    const float MAX_INPUT_M_IN_PASS = select(push.spatial_reuse_pass_idx == 0, MAX_INPUT_M_IN_PASS0, MAX_INPUT_M_IN_PASS1);

    // TODO: consider keeping high in areas of high variance.
    if (RTDGI_RESTIR_SPATIAL_USE_KERNEL_NARROWING) {
        kernel_tightness = mix(
            kernel_tightness, 1.0,
            0.5 * smoothstep(MAX_INPUT_M_IN_PASS * 0.5, MAX_INPUT_M_IN_PASS, center_r.M));
    }

    float max_kernel_radius =
        select(push.spatial_reuse_pass_idx == 0, mix(32.0, 12.0, kernel_tightness), mix(16.0, 6.0, kernel_tightness));

    // TODO: only run more passes where absolutely necessary (dispatch in tiles)
    if (push.spatial_reuse_pass_idx >= 2) {
        max_kernel_radius = 8;
    }

    const vec2 dist_to_edge_xy = min(vec2(px), push.output_tex_size.xy - px);
    const float allow_edge_overstep = select(center_r.M < 10, 100.0, 1.25);
    // const float allow_edge_overstep = 1.25;
    const vec2 kernel_radius = min(vec2(max_kernel_radius), dist_to_edge_xy * allow_edge_overstep);
    // const vec2 kernel_radius = max_kernel_radius;

    uint sample_count = select(DIFFUSE_GI_USE_RESTIR != 0, select(push.spatial_reuse_pass_idx == 0, SAMPLE_COUNT_PASS0, SAMPLE_COUNT_PASS1), 1);

#if 1
    // Scrambling angles here would be nice, but results in bad cache thrashing.
    // Quantizing the offsets results in mild cache abuse, and fixes most of the artifacts
    // (flickering near edges, e.g. under sofa in the UE5 archviz apartment scene).
    const uvec2 ang_offset_seed = select(bvec2(push.spatial_reuse_pass_idx == 0), (px >> 3), (px >> 2));
#else
    // Haha, cache go brrrrrrr.
    const uvec2 ang_offset_seed = px;
#endif

    float ang_offset = uint_to_u01_float(hash3(
                           uvec3(ang_offset_seed, deref(gpu_input).frame_index * 2 + push.spatial_reuse_pass_idx))) *
                       M_PI * 2;

    if (!RESTIR_USE_SPATIAL) {
        sample_count = 1;
    }

    vec3 radiance_output = vec3(0);

    for (uint sample_i = 0; sample_i < sample_count; ++sample_i) {
        // float ang = M_PI / 2;
        float ang = (sample_i + ang_offset) * GOLDEN_ANGLE;
        vec2 radius =
            select(bvec2(0 == sample_i), vec2(0), (pow(float(sample_i + sample_radius_offset) / sample_count, 0.5) * kernel_radius));
        ivec2 rpx_offset = ivec2(vec2(cos(ang), sin(ang)) * radius);

        const bool is_center_sample = sample_i == 0;
        // const bool is_center_sample = all(rpx_offset == 0);

        const ivec2 rpx = ivec2(px) + rpx_offset;

        const uvec2 reservoir_raw = safeTexelFetchU(reservoir_input_tex, ivec2(rpx), 0).xy;
        if (0 == reservoir_raw.x) {
            // Invalid reprojectoin
            continue;
        }

        Reservoir1spp r = Reservoir1spp_from_raw(reservoir_raw);

        r.M = min(r.M, 500);

        const uvec2 spx = reservoir_payload_to_px(r.payload);

        const TemporalReservoirOutput spx_packed = TemporalReservoirOutput_from_raw(safeTexelFetchU(temporal_reservoir_packed_tex, ivec2(spx), 0));
        const float reused_luminance = spx_packed.luminance;

        float visibility = 1;
        float relevance = 1;

        // Note: we're using `rpx` (neighbor reservoir px) here instead of `spx` (original ray px),
        // since we're merging with the stream of the neighbor and not the original ray.
        //
        // The distinction is in jacobians -- during every exchange, they get adjusted so that the target
        // pixel has correctly distributed rays. If we were to merge with the original pixel's stream,
        // we'd be applying the reservoirs several times.
        //
        // Consider for example merging a pixel with itself (no offset) multiple times over; we want
        // the jacobian to be 1.0 in that case, and not to reflect wherever its ray originally came from.

        const ivec2 sample_offset = ivec2(px) - ivec2(rpx);
        const float sample_dist2 = dot(vec2(sample_offset), vec2(sample_offset));
        const vec3 sample_normal_vs = safeTexelFetch(half_view_normal_tex, ivec2(rpx), 0).rgb;

        vec3 sample_radiance;
        if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
            sample_radiance = safeTexelFetch(bounced_radiance_input_tex, ivec2(rpx), 0).rgb;
        }

        const float normal_similarity_dot = dot(sample_normal_vs, center_normal_vs);
#if ALLOW_REUSE_OF_BACKFACING
        // Allow reuse even with surfaces that face away, but weigh them down.
        relevance *= normal_inluence_nonlinearity(normal_similarity_dot, 0.5) / normal_inluence_nonlinearity(1.0, 0.5);
#else
        relevance *= max(0, normal_similarity_dot);
#endif

        const float sample_ssao = safeTexelFetch(half_ssao_tex, ivec2(rpx), 0).r;

#if USE_SSAO_WEIGHING
        relevance *= 1 - abs(sample_ssao - center_ssao);
#endif

        const vec2 rpx_uv = get_uv(
            rpx * 2 + HALFRES_SUBSAMPLE_OFFSET,
            push.gbuffer_tex_size);
        const float rpx_depth = safeTexelFetch(half_depth_tex, ivec2(rpx), 0).r;

        if (rpx_depth == 0.0) {
            continue;
        }

        const ViewRayContext rpx_ray_ctx = vrc_from_uv_and_depth(globals, rpx_uv, rpx_depth);

        const vec2 spx_uv = get_uv(
            spx * 2 + HALFRES_SUBSAMPLE_OFFSET,
            push.gbuffer_tex_size);
        const ViewRayContext spx_ray_ctx = vrc_from_uv_and_depth(globals, spx_uv, spx_packed.depth);
        const vec3 sample_hit_ws = spx_packed.ray_hit_offset_ws + ray_hit_ws(spx_ray_ctx);

        const vec3 reused_dir_to_sample_hit_unnorm_ws = sample_hit_ws - ray_hit_ws(rpx_ray_ctx);

        // const float reused_luminance = sample_hit_ws_and_luminance.a;

        // Note: we want the neighbor's sample, which might have been resampled already.
        const float reused_dist = length(reused_dir_to_sample_hit_unnorm_ws);
        const vec3 reused_dir_to_sample_hit_ws = reused_dir_to_sample_hit_unnorm_ws / reused_dist;

        const vec3 dir_to_sample_hit_unnorm = sample_hit_ws - ray_hit_ws(view_ray_context);
        const float dist_to_sample_hit = length(dir_to_sample_hit_unnorm);
        const vec3 dir_to_sample_hit = normalize(dir_to_sample_hit_unnorm);

        // Reject neighbors with vastly different depths
        if (!is_center_sample) {
            // Clamp the normal_vs.z so that we don't get arbitrarily loose depth comparison at grazing angles.
            const float depth_diff = abs(max(0.3, center_normal_vs.z) * (center_depth / rpx_depth - 1.0));

            const float depth_threshold =
                select(push.spatial_reuse_pass_idx == 0, 0.15, 0.1);

            relevance *= 1 - smoothstep(0.0, depth_threshold, depth_diff);
        }

        // Raymarch to check occlusion
        if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH && push.perform_occlusion_raymarch != 0) {
            const vec2 ray_orig_uv = spx_uv;

            // const float surface_offset_len = length(spx_ray_ctx.ray_hit_vs() - view_ray_context.ray_hit_vs());
            const float surface_offset_len = length(
                // Use the center depth for simplicity; this doesn't need to be exact.
                // Faster, looks about the same.
                ray_hit_vs(vrc_from_uv_and_depth(globals, ray_orig_uv, depth)) - ray_hit_vs(view_ray_context));

            // Multiplier over the surface offset from the center to the neighbor
            const float MAX_RAYMARCH_DIST_MULT = 3.0;

            // Trace towards the hit point.

            const vec3 raymarch_dir_unnorm_ws = sample_hit_ws - ray_hit_ws(view_ray_context);
            const vec3 raymarch_end_ws =
                ray_hit_ws(view_ray_context)
                // TODO: what's a good max distance to raymarch?
                + raymarch_dir_unnorm_ws * min(1.0, MAX_RAYMARCH_DIST_MULT * surface_offset_len / length(raymarch_dir_unnorm_ws));

            OcclusionScreenRayMarch raymarch = OcclusionScreenRayMarch_create(
                uv, view_ray_context.ray_hit_cs.xyz, ray_hit_ws(view_ray_context),
                raymarch_end_ws,
                push.gbuffer_tex_size.xy);
            with_max_sample_count(raymarch, 6);
            with_halfres_depth(raymarch, push.output_tex_size.xy, half_depth_tex);

            if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
                with_color_bounce(raymarch, reprojected_gi_tex);
            }

            march(gpu_input, globals, raymarch, visibility, sample_radiance);
        }

        const vec3 sample_hit_normal_ws = spx_packed.hit_normal_ws;

        // phi_2^r in the ReSTIR GI paper
        const float center_to_hit_vis = -dot(sample_hit_normal_ws, dir_to_sample_hit);

        // phi_2^q
        const float reused_to_hit_vis = -dot(sample_hit_normal_ws, reused_dir_to_sample_hit_ws);

        float p_q = 1;
        if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
            p_q *= sRGB_to_luminance(sample_radiance);
        } else {
            p_q *= reused_luminance;
        }

        // Unlike in temporal reuse, here we can (and should) be running this.
        p_q *= max(0, dot(dir_to_sample_hit, center_normal_ws));

        float jacobian = 1;

        // Distance falloff. Needed to avoid leaks.
        jacobian *= reused_dist / dist_to_sample_hit;
        jacobian *= jacobian;

        // N of hit dot -L. Needed to avoid leaks. Without it, light "hugs" corners.
        //
        // Note: importantly, using the neighbor's data, not the original ray.
        jacobian *= clamp(center_to_hit_vis / reused_to_hit_vis, 0, 1e4);

        // Clearly wrong, but!:
        // The Jacobian introduces additional noise in corners, which is difficult to filter.
        // We still need something _resembling_ the jacobian in order to get directional cutoff,
        // and avoid leaks behind surfaces, but we don't actually need the precise Jacobian.
        // This causes us to lose some energy very close to corners, but with the near field split,
        // we don't need it anyway -- and it's better not to have the larger dark halos near corners,
        // which fhe full jacobian can cause due to imperfect integration (color bbox filters, etc).
        jacobian = sqrt(jacobian);

        if (is_center_sample) {
            jacobian = 1;
        }

        // Clamp neighbors give us a hit point that's considerably easier to sample
        // from our own position than from the neighbor. This can cause some darkening,
        // but prevents fireflies.
        //
        // The darkening occurs in corners, where micro-bounce should be happening instead.

        if (RTDGI_RESTIR_USE_JACOBIAN_BASED_REJECTION) {
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

        if (!(p_q >= 0)) {
            continue;
        }

        r.M *= relevance;

        if (push.occlusion_raymarch_importance_only != 0) {
            // This is used with ray-traced reservoir visibility which happens after
            // the last spatial resampling. We don't _need_ to perform the raymarch
            // for it, but importance sampling based on unshadowed contribution
            // could end up choosing occluded areas, which then get turned black
            // by the ray-traced check. This then creates extra variance.
            //
            // We can instead try to use the ray-marched visibility as an estimator
            // of real visibility.

            p_q *= mix(0.25, 1.0, visibility);
            visibility = 1;
        }

        if (update_with_stream(reservoir,
                               r, p_q, visibility * jacobian,
                               stream_state, r.payload, rng)) {
            dir_sel = dir_to_sample_hit;
            radiance_output = sample_radiance;
        }
    }

    finish_stream(reservoir, stream_state);
    reservoir.W = min(reservoir.W, RESTIR_RESERVOIR_W_CLAMP);

    safeImageStoreU(reservoir_output_tex, ivec2(px), uvec4(as_raw(reservoir), 0, 0));

    if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
        safeImageStore(bounced_radiance_output_tex, ivec2(px), vec4(radiance_output, 0.0));
    }
}

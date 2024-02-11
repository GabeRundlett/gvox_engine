#include <renderer/kajiya/rtr.inl>

#include <utilities/gpu/camera.glsl>
#include <utilities/gpu/rt.glsl>
// #include <utilities/gpu/uv.glsl>
// #include <utilities/gpu/pack_unpack.glsl>
// #include <utilities/gpu/frame_constants.glsl>
#include <utilities/gpu/gbuffer.glsl>
#include <utilities/gpu/brdf.glsl>
#include <utilities/gpu/brdf_lut.glsl>
#include <utilities/gpu/layered_brdf.glsl>
#include "blue_noise.glsl"
// #include <utilities/gpu/atmosphere.glsl>
// #include <utilities/gpu/sun.glsl>
// #include <utilities/gpu/lights/triangle.glsl>
#include <utilities/gpu/reservoir.glsl>
#include "rtr_settings.glsl"
#include "rtr_restir_pack_unpack.inc.glsl"
#include <utilities/gpu/downscale.glsl>
#include <utilities/gpu/safety.glsl>

#define RESTIR_RESERVOIR_W_CLAMP 1e20
#define RTR_RESTIR_BRDF_SAMPLING 1
const bool USE_SPATIAL_TAPS_AT_LOW_M = true;
const bool USE_RESAMPLING = true;

// Reject where the ray origin moves a lot
const bool USE_TRANSLATIONAL_CLAMP = true;

// Fixes up some ellipses near contacts
const bool USE_JACOBIAN_BASED_REJECTION = true;

// Causes some energy loss near contacts, but prevents
// ReSTIR from over-obsessing over them, and rendering
// tiny circles close to surfaces.
//
// TODO: This problem seems somewhat similar to what MIS fixes
// for light sampling; ReSTIR here is similar in behavior to a light sampling technique,
// and it similarly becomes bad close to the source, where BRDF sampling
// works perfectly fine. Maybe we can tackle it in a similar way.
const bool USE_DISTANCE_BASED_M_CLAMP = false;

const bool USE_REPROJECTION_SEARCH = true;

#define max_3(x, y, z) max(x, max(y, z))

DAXA_DECL_PUSH_CONSTANT(RtrRestirTemporalComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewIndex gbuffer_tex = push.uses.gbuffer_tex;
daxa_ImageViewIndex half_view_normal_tex = push.uses.half_view_normal_tex;
daxa_ImageViewIndex depth_tex = push.uses.depth_tex;
daxa_ImageViewIndex candidate0_tex = push.uses.candidate0_tex;
daxa_ImageViewIndex candidate1_tex = push.uses.candidate1_tex;
daxa_ImageViewIndex candidate2_tex = push.uses.candidate2_tex;
daxa_ImageViewIndex irradiance_history_tex = push.uses.irradiance_history_tex;
daxa_ImageViewIndex ray_orig_history_tex = push.uses.ray_orig_history_tex;
daxa_ImageViewIndex ray_history_tex = push.uses.ray_history_tex;
daxa_ImageViewIndex rng_history_tex = push.uses.rng_history_tex;
daxa_ImageViewIndex reservoir_history_tex = push.uses.reservoir_history_tex;
daxa_ImageViewIndex reprojection_tex = push.uses.reprojection_tex;
daxa_ImageViewIndex hit_normal_history_tex = push.uses.hit_normal_history_tex;
daxa_ImageViewIndex irradiance_out_tex = push.uses.irradiance_out_tex;
daxa_ImageViewIndex ray_orig_output_tex = push.uses.ray_orig_output_tex;
daxa_ImageViewIndex ray_output_tex = push.uses.ray_output_tex;
daxa_ImageViewIndex rng_output_tex = push.uses.rng_output_tex;
daxa_ImageViewIndex hit_normal_output_tex = push.uses.hit_normal_output_tex;
daxa_ImageViewIndex reservoir_out_tex = push.uses.reservoir_out_tex;

const float SKY_DIST = 1e4;

uvec2 reservoir_payload_to_px(uint payload) {
    return uvec2(payload & 0xffff, payload >> 16);
}

struct TraceResult {
    vec3 out_value;
    vec3 hit_normal_ws;
    vec3 hit_vs;
    float hit_t;
    float pdf;
    float cos_theta;
};

TraceResult do_the_thing(uvec2 px, vec3 primary_hit_normal) {
    const vec4 hit0 = safeTexelFetch(candidate0_tex, ivec2(px), 0);
    const vec4 hit1 = safeTexelFetch(candidate1_tex, ivec2(px), 0);
    const vec4 hit2 = safeTexelFetch(candidate2_tex, ivec2(px), 0);

    TraceResult result;
    result.out_value = hit0.rgb;
    result.pdf = min(hit1.a, RTR_RESTIR_MAX_PDF_CLAMP);
    result.cos_theta = rtr_decode_cos_theta_from_fp16(hit0.a);
    result.hit_vs = hit1.xyz;
    result.hit_t = length(hit1.xyz);
    result.hit_normal_ws = direction_view_to_world(globals, hit2.xyz);
    return result;
}

vec4 decode_hit_normal_and_dot(vec4 val) {
    return vec4(val.xyz * 2 - 1, val.w);
}

vec4 encode_hit_normal_and_dot(vec4 val) {
    return vec4(val.xyz * 0.5 + 0.5, val.w);
}

// Sometimes the best reprojection of the point-sampled reservoir data is not exactly at the pixel we're looking at.
// We need to inspect a tiny neighborhood around the point. This helps with shimmering edges, but also upon
// movement, since we can't linearly sample the reservoir data.
void find_best_reprojection_in_neighborhood(vec2 base_px, inout ivec2 best_px, vec3 refl_ray_origin_ws, bool wide) {
    float best_dist = 1e10;

    const vec2 clip_scale = vec2(deref(globals).player.cam.clip_to_view[0][0], deref(globals).player.cam.clip_to_view[1][1]);
    const vec2 offset_scale = vec2(1, -1) * -2 * clip_scale * push.gbuffer_tex_size.zw;

    const vec3 look_direction = direction_view_to_world(globals, vec3(0, 0, -1));

    {
        const float z_offset = dot(look_direction, refl_ray_origin_ws - get_eye_position(globals));

        // Subtract the subsample XY offset from the comparison position.
        // This will prevent the search from constantly re-shuffling pixels due to the sub-sample jitters.
        refl_ray_origin_ws += direction_view_to_world(globals, vec3(vec2(HALFRES_SUBSAMPLE_OFFSET) * offset_scale * z_offset, 0));
    }

    const int start_coord = select(wide, -1, 0);
    for (int y = start_coord; y <= 1; ++y) {
        for (int x = start_coord; x <= 1; ++x) {
            ivec2 spx = ivec2(floor(base_px + vec2(x, y)));

            RtrRestirRayOrigin ray_orig = RtrRestirRayOrigin_from_raw(safeTexelFetch(ray_orig_history_tex, ivec2(spx), 0));
            vec3 orig = ray_orig.ray_origin_eye_offset_ws + get_prev_eye_position(globals);
            uvec2 orig_jitter = hi_px_subpixels[ray_orig.frame_index_mod4];

            {
                const float z_offset = dot(look_direction, orig);

                // Similarly subtract the subsample XY offset that the ray was traced with.
                orig += direction_view_to_world(globals, vec3(vec2(orig_jitter) * offset_scale * z_offset, 0));
            }

            float d = length(orig - refl_ray_origin_ws);

            if (d < best_dist) {
                best_dist = d;
                best_px = spx;
            }
        }
    }
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    const uvec2 px = gl_GlobalInvocationID.xy;
    const uvec2 hi_px_offset = HALFRES_SUBSAMPLE_OFFSET;
    const uvec2 hi_px = px * 2 + hi_px_offset;

    float depth = safeTexelFetch(depth_tex, ivec2(hi_px), 0).r;

    if (0.0 == depth) {
        safeImageStore(irradiance_out_tex, ivec2(px), vec4(0.0.xxx, -SKY_DIST));
        safeImageStore(hit_normal_output_tex, ivec2(px), 0.0.xxxx);
        safeImageStoreU(reservoir_out_tex, ivec2(px), uvec4(0));
        return;
    }

    const vec2 uv = get_uv(hi_px, push.gbuffer_tex_size);
    const vec3 normal_vs = safeTexelFetch(half_view_normal_tex, ivec2(px), 0).xyz;
    const vec3 normal_ws = direction_view_to_world(globals, normal_vs);

    float local_normal_flatness = 1;
    {
        const int k = 1;
        for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                vec3 sn_vs = safeTexelFetch(half_view_normal_tex, ivec2(px + ivec2(x, y)), 0).xyz;
                local_normal_flatness *= saturate(dot(normal_vs, sn_vs));
            }
        }
    }

    float reprojection_neighborhood_stability = 1;
    {
        for (int y = 0; y <= 1; ++y) {
            for (int x = 0; x <= 1; ++x) {
                float r = safeTexelFetch(reprojection_tex, ivec2(px * 2 + ivec2(x, y)), 0).z;
                reprojection_neighborhood_stability *= r;
            }
        }
    }

#if RTR_USE_TIGHTER_RAY_BIAS
    const ViewRayContext view_ray_context = vrc_from_uv_and_biased_depth(globals, uv, depth);
    const vec3 refl_ray_origin_ws = biased_secondary_ray_origin_ws_with_normal(view_ray_context, normal_ws);
#else
    const ViewRayContext view_ray_context = vrc_from_uv_and_depth(globals, uv, depth);
    const vec3 refl_ray_origin_ws = biased_secondary_ray_origin_ws(view_ray_context);
#endif

    const vec3 refl_ray_origin_vs = position_world_to_view(globals, refl_ray_origin_ws);

    const mat3 tangent_to_world = build_orthonormal_basis(normal_ws);
    vec3 outgoing_dir = vec3(0, 0, 1);

    uint rng = hash3(uvec3(px, deref(gpu_input).frame_index));
    vec3 wo = (-normalize(ray_dir_ws(view_ray_context))) * tangent_to_world;

    // Hack for shading normals facing away from the outgoing ray's direction:
    // We flip the outgoing ray along the shading normal, so that the reflection's curvature
    // continues, albeit at a lower rate.
    if (wo.z < 0.0) {
        wo.z *= -0.25;
        wo = normalize(wo);
    }

    GbufferData gbuffer = unpack(GbufferDataPacked(safeTexelFetchU(gbuffer_tex, ivec2(hi_px), 0)));
    SpecularBrdf specular_brdf;
    {
        LayeredBrdf layered_brdf = LayeredBrdf_from_gbuffer_ndotv(gbuffer, wo.z);
        specular_brdf = layered_brdf.specular_brdf;
    }
    const float a2 = max(RTR_ROUGHNESS_CLAMP, gbuffer.roughness) * max(RTR_ROUGHNESS_CLAMP, gbuffer.roughness);

    // TODO: use
    vec3 light_radiance = 0.0.xxx;

    float p_q_sel = 0;
    float pdf_sel = 0;
    float cos_theta = 0;
    vec3 irradiance_sel = vec3(0);
    vec4 ray_orig_sel = vec4(0);
    vec3 ray_hit_sel_ws = vec3(1);
    vec3 hit_normal_sel = vec3(1);
    uint rng_sel = safeTexelFetchU(rng_output_tex, ivec2(px), 0).r;

    Reservoir1sppStreamState stream_state = Reservoir1sppStreamState_create();
    Reservoir1spp reservoir = Reservoir1spp_create();
    const uint reservoir_payload = px.x | (px.y << 16);

    reservoir.payload = reservoir_payload;

    {
        TraceResult result = do_the_thing(px, normal_ws);

        if (result.pdf > 0) {
            outgoing_dir = normalize(result.hit_vs);

            vec3 wi = normalize(outgoing_dir * tangent_to_world);

            const float p_q = p_q_sel = 1 * max(1e-3, sRGB_to_luminance(result.out_value))
#if !RTR_RESTIR_BRDF_SAMPLING
                                        * max(0, dot(outgoing_dir, normal_ws))
#endif
                                        //* sRGB_to_luminance(specular_brdf.evaluate(wo, wi).value)
                                        * result.pdf;

            const float inv_pdf_q = 1.0 / result.pdf;

            pdf_sel = result.pdf;
            cos_theta = result.cos_theta;

            irradiance_sel = result.out_value;

            RtrRestirRayOrigin ray_orig;
            // Note: needs patching up by the eye pos later.
            ray_orig.ray_origin_eye_offset_ws = refl_ray_origin_ws;
            ray_orig.roughness = gbuffer.roughness;
            ray_orig.frame_index_mod4 = deref(gpu_input).frame_index & 3;

            ray_orig_sel = to_raw(ray_orig);

            ray_hit_sel_ws = result.hit_vs + refl_ray_origin_ws;

            hit_normal_sel = result.hit_normal_ws;

            if (p_q * inv_pdf_q > 0) {
                init_with_stream(reservoir, p_q, inv_pdf_q, stream_state, reservoir_payload);
            }
        }
    }

    // const bool use_resampling = false;
    const bool use_resampling = USE_RESAMPLING;
    const vec4 center_reproj = safeTexelFetch(reprojection_tex, ivec2(hi_px), 0);

    if (use_resampling) {
        const float ang_offset = ((deref(gpu_input).frame_index + 7) * 11) % 32 * M_TAU;

        for (uint sample_i = 0; sample_i < select((USE_SPATIAL_TAPS_AT_LOW_M && center_reproj.z < 1.0), 5, 1) && stream_state.M_sum < RTR_RESTIR_TEMPORAL_M_CLAMP; ++sample_i) {
            // for (uint sample_i = 0; sample_i < 1; ++sample_i) {
            const float ang = (sample_i + ang_offset) * GOLDEN_ANGLE;
            const float rpx_offset_radius = sqrt(
                                                float(((sample_i - 1) + deref(gpu_input).frame_index) & 3) + 1) *
                                            clamp(8 - stream_state.M_sum, 1, 7); // TODO: keep high in noisy situations
            //) * 7;
            const vec2 reservoir_px_offset_base = vec2(cos(ang), sin(ang)) * rpx_offset_radius;

            const ivec2 rpx_offset =
                select(bvec2(sample_i == 0), ivec2(0, 0), ivec2(reservoir_px_offset_base));

            vec4 reproj = safeTexelFetch(reprojection_tex, ivec2(hi_px + rpx_offset * 2), 0);

            // Can't use linear interpolation.

            const vec2 reproj_px_flt = px + push.gbuffer_tex_size.xy * reproj.xy / 2;

            ivec2 reproj_px;

            {
                const vec2 base_px = px + push.gbuffer_tex_size.xy * reproj.xy / 2;
                ivec2 best_px = ivec2(floor(base_px + 0.5));

                if (USE_REPROJECTION_SEARCH) {
#if USE_HALFRES_SUBSAMPLE_JITTERING
                    if (reprojection_neighborhood_stability >= 1) {
                        // If the neighborhood is stable, we can do a tiny search to find a reprojection
                        // that has the best chance of keeping reservoirs alive.
                        // Only bother if there's any motion. If we do this when there's no motion,
                        // we may end up creating too much correlation between pixels by shuffling them around.

                        if (any(greaterThan(abs(push.gbuffer_tex_size.xy * reproj.xy), vec2(0.1)))) {
                            find_best_reprojection_in_neighborhood(base_px, best_px, refl_ray_origin_ws.xyz, false);
                        }
                    } else {
                        // The neighborhood is not stable. Shimmering or moving edges.
                        // Do a more aggressive search.

                        find_best_reprojection_in_neighborhood(base_px, best_px, refl_ray_origin_ws.xyz, true);
                    }
#else
                    // If subsample jittering is disabled, we only ever need the tiny search

                    if (any(greaterThan(abs(push.gbuffer_tex_size.xy * reproj.xy), vec2(0.1)))) {
                        find_best_reprojection_in_neighborhood(base_px, best_px, refl_ray_origin_ws.xyz, false);
                    }
#endif
                }

                reproj_px = best_px;
            }

            const ivec2 rpx = reproj_px + rpx_offset;
            const uvec2 rpx_hi = rpx * 2 + hi_px_offset;

            const vec3 sample_normal_vs = safeTexelFetch(half_view_normal_tex, ivec2(rpx), 0).xyz;
            // Note: also doing this for sample 0, as under extreme aliasing,
            // we can easily get bad samples in.
            if (dot(sample_normal_vs, normal_vs) < 0.7) {
                // continue;
            }

            Reservoir1spp r = Reservoir1spp_from_raw(safeTexelFetchU(reservoir_history_tex, ivec2(rpx), 0).xy);
            const uvec2 spx = reservoir_payload_to_px(r.payload);

            const vec2 sample_uv = get_uv(rpx_hi, push.gbuffer_tex_size);
            const vec4 prev_ray_orig_and_roughness = safeTexelFetch(ray_orig_history_tex, ivec2(spx), 0) + vec4(get_prev_eye_position(globals), 0);

            // Reject disocclusions
            if (length_squared(refl_ray_origin_ws - prev_ray_orig_and_roughness.xyz) > 0.05 * refl_ray_origin_vs.z * refl_ray_origin_vs.z) {
                continue;
            }

            const vec4 prev_irrad_and_cos_theta =
                safeTexelFetch(irradiance_history_tex, ivec2(spx), 0) * vec4((deref(gpu_input).pre_exposure_delta).xxx, 1);

            const vec3 prev_irrad = prev_irrad_and_cos_theta.rgb;
            const float prev_cos_theta = rtr_decode_cos_theta_from_fp16(prev_irrad_and_cos_theta.a);

            const vec4 sample_hit_ws_and_pdf_packed = safeTexelFetch(ray_history_tex, ivec2(spx), 0);
            const float prev_pdf = sample_hit_ws_and_pdf_packed.a;

            const vec3 sample_hit_ws = sample_hit_ws_and_pdf_packed.xyz + prev_ray_orig_and_roughness.xyz;
            // const vec3 prev_dir_to_sample_hit_unnorm_ws = sample_hit_ws - sample_ray_ctx.ray_hit_ws();
            // const vec3 prev_dir_to_sample_hit_ws = normalize(prev_dir_to_sample_hit_unnorm_ws);
            const float prev_dist = length(sample_hit_ws_and_pdf_packed.xyz);
            // const float prev_dist = length(prev_dir_to_sample_hit_unnorm_ws);

            // Note: needs `spx` since `hit_normal_history_tex` is not reprojected.
            const vec4 sample_hit_normal_ws_dot = decode_hit_normal_and_dot(safeTexelFetch(hit_normal_history_tex, ivec2(spx), 0));

            const vec3 dir_to_sample_hit_unnorm = sample_hit_ws - refl_ray_origin_ws;
            const float dist_to_sample_hit = length(dir_to_sample_hit_unnorm);
            const vec3 dir_to_sample_hit = normalize(dir_to_sample_hit_unnorm);

            // From the ReSTIR paper:
            // With temporal reuse, the number of candidates M contributing to the
            // pixel can in theory grow unbounded, as each frame always combines
            // its reservoir with the previous frame’s. This causes (potentially stale)
            // temporal samples to be weighted disproportionately high during
            // resampling. To fix this, we simply clamp the previous frame’s M
            // to at most 20× of the current frame’s reservoir’s M

            r.M = min(r.M, RTR_RESTIR_TEMPORAL_M_CLAMP);

            const vec3 wi = normalize(dir_to_sample_hit * tangent_to_world);

            if (USE_TRANSLATIONAL_CLAMP) {
                // const vec3 current_wo = normalize(ViewRayContext::from_uv(uv).ray_dir_vs());
                // const vec3 prev_wo = normalize(ViewRayContext::from_uv(uv + center_reproj.xy).ray_dir_vs());

                // TODO: take object motion into account too
                const vec3 current_wo = normalize(ray_hit_ws(view_ray_context) - get_eye_position(globals));
                const vec3 prev_wo = normalize(ray_hit_ws(view_ray_context) - get_prev_eye_position(globals));

                const float wo_dot = saturate(dot(current_wo, prev_wo));

                const float wo_similarity =
                    pow(saturate(SpecularBrdf_ggx_ndf_0_1(max(3e-5, a2), wo_dot)), 64);

                float mult = mix(wo_similarity, 1, smoothstep(0.05, 0.5, sqrt(gbuffer.roughness)));

                // Don't bother if the surface is bumpy. The lag is hard to see then,
                // and we'd just end up introducing aliasing on small features.
                mult = mix(1.0, mult, local_normal_flatness);

                r.M *= mult;
            }

            float p_q = 1;
            p_q *= max(1e-3, sRGB_to_luminance(prev_irrad.rgb));
#if !RTR_RESTIR_BRDF_SAMPLING
            p_q *= max(0, dot(dir_to_sample_hit, normal_ws));
#else
            p_q *= step(0, dot(dir_to_sample_hit, normal_ws));
#endif
            p_q *= prev_pdf;
            // p_q *= sRGB_to_luminance(specular_brdf.evaluate(wo, wi).value);

            float visibility = 1;
            float jacobian = 1;

            // Note: needed for sample 0 due to temporal jitter.
            if (true) {
                // Distance falloff. Needed to avoid leaks.
                jacobian *= clamp(prev_dist / dist_to_sample_hit, 1e-4, 1e4);
                jacobian *= jacobian;

                // N of hit dot -L. Needed to avoid leaks.
                jacobian *=
                    max(0.0, -dot(sample_hit_normal_ws_dot.xyz, dir_to_sample_hit)) / max(1e-5, sample_hit_normal_ws_dot.w);
                /// max(1e-5, -dot(sample_hit_normal_ws_dot.xyz, prev_dir_to_sample_hit_ws));

#if RTR_RESTIR_BRDF_SAMPLING
                // N dot L. Useful for normal maps, micro detail.
                // The min(const, _) should not be here, but it prevents fireflies and brightening of edges
                // when we don't use a harsh normal cutoff to exchange reservoirs with.
                // jacobian *= min(1.2, max(0.0, prev_irrad.a) / dot(dir_to_sample_hit, center_normal_ws));
                // jacobian *= max(0.0, prev_irrad.a) / dot(dir_to_sample_hit, center_normal_ws);
#endif
            }

            // Fixes boiling artifacts near edges. Unstable jacobians,
            // but also effectively reduces reliance on reservoir exchange
            // in tight corners, which is desirable since the well-distributed
            // raw samples thrown at temporal filters will do better.
            if (USE_JACOBIAN_BASED_REJECTION) {
                const float JACOBIAN_REJECT_THRESHOLD = mix(1.1, 4.0, gbuffer.roughness * gbuffer.roughness);
                if (!(jacobian < JACOBIAN_REJECT_THRESHOLD && jacobian > 1.0 / JACOBIAN_REJECT_THRESHOLD)) {
                    continue;
                    // r.M *= pow(saturate(1 - max(jacobian, 1.0 / jacobian) / JACOBIAN_REJECT_THRESHOLD), 4.0);
                }
            }

            if (USE_DISTANCE_BASED_M_CLAMP) {
                // ReSTIR tends to produce firflies near contacts.
                // This is a hack to reduce the effect while I figure out a better solution.
                // HACK: reduce M close to surfaces.
                //
                // Note: This causes ReSTIR to be less effective, and can manifest
                // as darkening in corners. Since it's mostly useful for smoother surfaces,
                // fade it out when they're rough.
                const float dist_to_hit_vs_scaled =
                    dist_to_sample_hit / -refl_ray_origin_vs.z * deref(globals).player.cam.view_to_clip[1][1];
                {
                    float dist2 = dot(ray_hit_sel_ws - refl_ray_origin_ws, ray_hit_sel_ws - refl_ray_origin_ws);
                    dist2 = min(dist2, 2 * dist_to_hit_vs_scaled * dist_to_hit_vs_scaled);
                    r.M = min(r.M, RTR_RESTIR_TEMPORAL_M_CLAMP * mix(saturate(50.0 * dist2), 1.0, gbuffer.roughness * gbuffer.roughness));
                }
            }

            // We're not recalculating the PDF-based factor of p_q,
            // so it needs measure adjustment.
            p_q *= jacobian;

            // TODO: consider ray-marching for occlusion

            if (update_with_stream(reservoir, r, p_q, visibility,
                                   stream_state, reservoir_payload, rng)) {
                outgoing_dir = dir_to_sample_hit;
                p_q_sel = p_q;
                pdf_sel = prev_pdf;
                cos_theta = prev_cos_theta;
                irradiance_sel = prev_irrad.rgb;

                // TODO: was `refl_ray_origin_ws`; what should it be?
                // ray_orig_sel = refl_ray_origin_ws;
                ray_orig_sel = prev_ray_orig_and_roughness;

                ray_hit_sel_ws = sample_hit_ws;
                hit_normal_sel = sample_hit_normal_ws_dot.xyz;

                rng_sel = safeTexelFetchU(rng_history_tex, ivec2(spx), 0).x;
            }
        }

        finish_stream(reservoir, stream_state);
        reservoir.W = min(reservoir.W, RESTIR_RESERVOIR_W_CLAMP);
    }

    RayDesc outgoing_ray;
    outgoing_ray.Direction = outgoing_dir;
    outgoing_ray.Origin = refl_ray_origin_ws;
    outgoing_ray.TMin = 0;

    const vec4 hit_normal_ws_dot = vec4(hit_normal_sel, -dot(hit_normal_sel, outgoing_ray.Direction));

    safeImageStore(irradiance_out_tex, ivec2(px), vec4(irradiance_sel, rtr_encode_cos_theta_for_fp16(cos_theta)));
    // Note: relies on the `xyz` being directly encoded by `RtrRestirRayOrigin`
    safeImageStore(ray_orig_output_tex, ivec2(px), vec4(ray_orig_sel.xyz - get_eye_position(globals), ray_orig_sel.w));
    safeImageStore(hit_normal_output_tex, ivec2(px), encode_hit_normal_and_dot(hit_normal_ws_dot));
    safeImageStore(ray_output_tex, ivec2(px), vec4(ray_hit_sel_ws - ray_orig_sel.xyz, pdf_sel));
    safeImageStoreU(rng_output_tex, ivec2(px), uvec4(rng_sel, 0, 0, 0));
    safeImageStoreU(reservoir_out_tex, ivec2(px), uvec4(as_raw(reservoir), 0, 0));
}

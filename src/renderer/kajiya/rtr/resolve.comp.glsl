#include <renderer/kajiya/rtr.inl>

#include <renderer/kajiya/inc/camera.glsl>
#include "../inc/color.glsl"
#include <g_samplers>
// #include "../inc/frame_constants.glsl"
// #include <utilities/gpu/pack_unpack.glsl>
#include "../inc/brdf.glsl"
#include "../inc/brdf_lut.glsl"
#include "../inc/layered_brdf.glsl"
// #include <utilities/gpu/uv.glsl>
// #include <utilities/gpu/hash.glsl>
#include "blue_noise.glsl"
#include "../inc/reservoir.glsl"
// #include <utilities/gpu/morton.glsl>
#include "../rtdgi/candidate_ray_dir.glsl"
#include "rtr_settings.glsl"
#include "rtr_restir_pack_unpack.inc.glsl"
#include "../inc/downscale.glsl"
#include "../inc/safety.glsl"

DAXA_DECL_PUSH_CONSTANT(RtrRestirResolveComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewIndex gbuffer_tex = push.uses.gbuffer_tex;
daxa_ImageViewIndex depth_tex = push.uses.depth_tex;
daxa_ImageViewIndex hit0_tex = push.uses.hit0_tex;
daxa_ImageViewIndex hit1_tex = push.uses.hit1_tex;
daxa_ImageViewIndex hit2_tex = push.uses.hit2_tex;
daxa_ImageViewIndex history_tex = push.uses.history_tex;
daxa_ImageViewIndex reprojection_tex = push.uses.reprojection_tex;
daxa_ImageViewIndex half_view_normal_tex = push.uses.half_view_normal_tex;
daxa_ImageViewIndex half_depth_tex = push.uses.half_depth_tex;
daxa_ImageViewIndex ray_len_history_tex = push.uses.ray_len_history_tex;
daxa_ImageViewIndex restir_irradiance_tex = push.uses.restir_irradiance_tex;
daxa_ImageViewIndex restir_ray_tex = push.uses.restir_ray_tex;
daxa_ImageViewIndex restir_reservoir_tex = push.uses.restir_reservoir_tex;
daxa_ImageViewIndex restir_ray_orig_tex = push.uses.restir_ray_orig_tex;
daxa_ImageViewIndex restir_hit_normal_tex = push.uses.restir_hit_normal_tex;
daxa_ImageViewIndex blue_noise_vec2 = push.uses.blue_noise_vec2;
daxa_ImageViewIndex output_tex = push.uses.output_tex;
daxa_ImageViewIndex ray_len_output_tex = push.uses.ray_len_output_tex;
daxa_ImageViewIndex rtr_debug_image = push.uses.rtr_debug_image;

// Skip some calculations if not visually relevant.
#define CUT_CORNERS_IN_MATH 1

const bool SHUFFLE_SUBPIXELS = true;
#define BORROW_SAMPLES 1
#define SHORT_CIRCUIT_NO_BORROW_SAMPLES 0

#define USE_APPROXIMATE_SAMPLE_SHADOWING 1

const uint MAX_SAMPLE_COUNT = 8;
const bool USE_RESTIR = RTR_USE_RESTIR;

float approx_fresnel(vec3 wo, vec3 wi) {
    vec3 h_unnorm = wo + wi;
    return exp2(-1.0 - 8.65617024533378 * dot(wi, h_unnorm));
}

bool is_wave_alive(uint mask, uint idx) {
    return (mask & (1u << idx)) != 0;
}

uvec2 reservoir_payload_to_px(uint payload) {
    return uvec2(payload & 0xffff, payload >> 16);
}

vec3 decode_restir_hit_normal(vec3 val) {
    return val.xyz * 2 - 1;
}

// Get tangent vectors for the basis for specular filtering.
// Based on "Fast Denoising with Self Stabilizing Recurrent Blurs" by Dmitry Zhdan
void get_specular_filter_kernel_basis(vec3 v, vec3 n, float roughness, float scale, out vec3 t1, out vec3 t2) {
    vec3 dominant = specular_dominant_direction(n, v, roughness);
    vec3 reflected = reflect(-dominant, n);

    t1 = normalize(cross(n, reflected)) * scale;
    t2 = cross(reflected, t1);
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    const uvec2 px = gl_GlobalInvocationID.xy;
    const uvec2 px_tile = gl_WorkGroupID.xy;
    const uint idx_within_group = gl_LocalInvocationIndex;
    // px = decode_morton_2d(idx_within_group) + px_tile * 8;
    // px = (px & ~3) | ((px & 1) << 1) | ((px & 2) >> 1);

    const uvec2 half_px = px / 2;

    const vec2 uv = get_uv(px, push.output_tex_size);
    const float depth = safeTexelFetch(depth_tex, ivec2(px), 0).r;

    if (0.0 == depth) {
        safeImageStore(output_tex, ivec2(px), vec4(0));
        return;
    }

#if !BORROW_SAMPLES && SHORT_CIRCUIT_NO_BORROW_SAMPLES
    safeImageStore(output_tex, ivec2(px), vec4(safeTexelFetch(hit0_tex, ivec2(half_px), 0).rgb, 0));
    safeImageStore(ray_len_output_tex, ivec2(px), vec4(1));
    return;
#endif

    GbufferData gbuffer = unpack(GbufferDataPacked(safeTexelFetchU(gbuffer_tex, ivec2(px), 0)));

#if RTR_USE_TIGHTER_RAY_BIAS
    const ViewRayContext view_ray_context = vrc_from_uv_and_biased_depth(gpu_input, uv, depth);
    const vec3 refl_ray_origin_ws = biased_secondary_ray_origin_ws_with_normal(view_ray_context, gbuffer.normal);
#else
    const ViewRayContext view_ray_context = vrc_from_uv_and_depth(gpu_input, uv, depth);
    const vec3 refl_ray_origin_ws = biased_secondary_ray_origin_ws(view_ray_context);
#endif

    const vec3 refl_ray_origin_vs = position_world_to_view(gpu_input, refl_ray_origin_ws);

    // Clamp to fix moire on mirror-like surfaces
    gbuffer.roughness = max(gbuffer.roughness, RTR_ROUGHNESS_CLAMP);

    const mat3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);
    vec3 wo = (-normalize(ray_dir_ws(view_ray_context))) * tangent_to_world;

    // Hack for shading normals facing away from the outgoing ray's direction:
    // We flip the outgoing ray along the shading normal, so that the reflection's curvature
    // continues, albeit at a lower rate.
    if (wo.z < 0.0) {
        wo.z *= -0.25;
        wo = normalize(wo);
    }

    SpecularBrdf specular_brdf;
    {
        LayeredBrdf layered_brdf = LayeredBrdf_from_gbuffer_ndotv(gbuffer, wo.z);
        specular_brdf = layered_brdf.specular_brdf;
    }
    const float f0_grey = sRGB_to_luminance(specular_brdf.albedo);

    // Index used to calculate a sample set disjoint for all four pixels in the quad
    // Offsetting by frame index reduces small structured artifacts
    const uint px_idx_in_quad = (((px.x & 1) | (px.y & 1) * 2) + select(SHUFFLE_SUBPIXELS, 1, 0) * deref(gpu_input).frame_index) & 3;

    const float a2 = max(RTR_ROUGHNESS_CLAMP, gbuffer.roughness) * max(RTR_ROUGHNESS_CLAMP, gbuffer.roughness);

    // Project lobe footprint onto the reflector, and find the desired convolution size
    const float surf_to_hit_dist = length(safeTexelFetch(hit1_tex, ivec2(half_px), 0).xyz);
    const float eye_to_surf_dist = length(refl_ray_origin_vs);

    // Needed to account for perspective distortion to keep the kernel constant
    // near screen boundaries at high FOV.
    const float eye_ray_z_scale = -ray_dir_vs(view_ray_context).z;

    const vec4 reprojection_params = safeTexelFetch(reprojection_tex, ivec2(px), 0);

    const float ray_squish_scale = 16.0 / max(1e-5, eye_to_surf_dist);
    const float ray_len_avg = exponential_unsquish(mix(
                                                       exponential_squish(textureLod(daxa_sampler2D(ray_len_history_tex, g_sampler_lnc), uv + reprojection_params.xy, 0).y, ray_squish_scale),
                                                       exponential_squish(surf_to_hit_dist, ray_squish_scale),
                                                       0.1),
                                                   ray_squish_scale);

    const uint sample_count = select(BORROW_SAMPLES != 0, MAX_SAMPLE_COUNT, 1);

    vec4 contrib_accum = vec4(0.0);
    vec2 w_accum = vec2(0.0);
    float ray_len_accum = 0;

    // float ex = 0.0;
    // float ex2 = 0.0;

    const vec3 normal_vs = direction_world_to_view(gpu_input, gbuffer.normal);

    // The footprint of our kernel increases with roughness, and scales with distance.
    // The latter is a simple relationship of distance to camera and distance to hit point,
    // and for the roughness factor we can bound a cone.
    //
    // According to 4-9-3-DistanceBasedRoughnessLobeBounding.nb
    // in "Moving Frostbite to PBR" by Seb and Charles, the cutoff angle tangent should be:
    // sqrt(energy_frac / (1.0 - energy_frac)) * roughness
    // That doesn't work however, resulting in a kernel which grows too slowly with
    // increasing roughness. Unclear why -- related to the cone being derived
    // for the NDF which exists in half-angle space?
    //
    // What does work however is a square root relationship (constants arbitrary):
    //
    // TODO: consider a pre-pass which analyzes the local neighborhood of every pixel,
    // finding an optimal radius for it, based on which samples will actually be useful.
    // It could return a radius, or directly produce sample offets packed in rgba32ui.
    const float tan_theta = sqrt(gbuffer.roughness) * 0.25;

    float kernel_size_ws;
    {
        // Clamp the ray length used in kernel size calculations, so we don't end up with
        // tiny kernels in corners. In the presense of fireflies (e.g. from reflected spec),
        // that would result in small circles appearing as reflections.
        float clamped_ray_len_avg = max(
            ray_len_avg,
            eye_to_surf_dist / eye_ray_z_scale * deref(gpu_input).player.cam.clip_to_view[1][1] * 0.2
                // Keep contacts sharp
                * smoothstep(0, 0.05 * eye_to_surf_dist, ray_len_avg));

        const float kernel_size_vs = clamped_ray_len_avg / (clamped_ray_len_avg + eye_to_surf_dist);
        kernel_size_ws = kernel_size_vs * eye_to_surf_dist * eye_ray_z_scale;
        kernel_size_ws *= tan_theta;
    }

    {
        const float scale_factor = eye_to_surf_dist * eye_ray_z_scale * deref(gpu_input).player.cam.clip_to_view[1][1];

        // Clamp the kernel size so we don't sample the same point, but also don't thrash all the caches.
        kernel_size_ws = min(kernel_size_ws, 0.1 * scale_factor);
        kernel_size_ws = max(kernel_size_ws, push.output_tex_size.w * 4.0 * scale_factor);
    }

    vec3 kernel_t1, kernel_t2;
    get_specular_filter_kernel_basis(
        -normalize(ray_dir_ws(view_ray_context)),
        gbuffer.normal,
        gbuffer.roughness,
        kernel_size_ws,
        kernel_t1, kernel_t2);

    // Offset to avoid correlation with ray generation
    vec4 blue = blue_noise_for_pixel(blue_noise_vec2, half_px + 16, deref(gpu_input).frame_index);

    // Feeds into the `pow` to remap sample index to radius.
    // At 0.5 (sqrt), it's disk sampling, with higher values becoming conical.
    // Must be constant, so the `pow` can be const-folded.
    const float KERNEL_SHARPNESS = 0.666;
    // const float KERNEL_SHARPNESS = 0.5;
    const float RADIUS_SAMPLE_MULT = 1.0 / pow(float(MAX_SAMPLE_COUNT), KERNEL_SHARPNESS);

    // const float ang_offset = blue.x;
    //  Way faster, seems to look the same.
    const float ang_offset = (deref(gpu_input).frame_index * 59 % 128) * M_PLASTIC;

    Reservoir1spp center_r = Reservoir1spp_from_raw(safeTexelFetchU(restir_reservoir_tex, ivec2(half_px), 0).xy);

    // Instead of directly increasing the sampling radius, we'll do something adaptive.
    // If a sample gets rejected, we only increase the radius of the next sample by a fraction
    // of what we would do otherwise. This keeps the radius from expanding too quickly
    // on small or thin elements, slightly reducing their noise.
    const float RADIUS_INC_ON_FAIL = 0.25;

    // We skip the first center sample initially, and then evaluate it last, using
    // information that we gather along the way to decide how to sample the center.
    float sample_radius_accum = 1;

    for (int sample_i = 1; sample_i <= sample_count; ++sample_i, sample_radius_accum += RADIUS_INC_ON_FAIL) {
        const bool is_center_sample = sample_i == sample_count;

        ivec2 sample_offset;
        {
            float ang = (sample_i + ang_offset) * GOLDEN_ANGLE + (px_idx_in_quad / 4.0) * M_TAU;
            float sample_i_with_jitter = sample_radius_accum;

            // If we've arrived at the central sample, and haven't found satisfactory
            // contributions yet, load the candidate directly from `half_px`.
            // Otherwise jitter the location to avoid half-pixel artifacts.
            // This additional check reduces undersampling on thin shiny surfaces,
            // which otherwise causes them to appear black.
            if (is_center_sample) {
                const bool offset_center_pixel = true
                                                 // roughness check is a HACK. it's only there because not offsetting
                                                 // the center pixel on rough surfaces causes pixellation.
                                                 // it can happen for thin features, or features seen at an angle.
                                                 //&& (contrib_accum.w > 1e-8 || gbuffer.roughness > 0.1)
                                                 && contrib_accum.w > 1e-8;

                if (offset_center_pixel) {
                    sample_i_with_jitter = blue.y;
                } else {
                    sample_i_with_jitter = 0;
                }
            } else {
                sample_i_with_jitter += blue.y;
            }

            const float radius = select(BORROW_SAMPLES, pow(sample_i_with_jitter, KERNEL_SHARPNESS) * RADIUS_SAMPLE_MULT, 0);

            vec3 offset_ws = (cos(ang) * kernel_t1 + sin(ang) * kernel_t2) * radius;
            vec3 sample_ws = refl_ray_origin_ws + offset_ws;
            vec3 sample_cs = position_world_to_sample(gpu_input, sample_ws);
            vec2 sample_uv = cs_to_uv(sample_cs.xy);

            // TODO: pass in `input_tex_size`
            ivec2 sample_px = ivec2(floor(sample_uv * push.output_tex_size.xy / 2));
            sample_offset = sample_px - ivec2(half_px);
        }

        const ivec2 sample_px = ivec2(half_px) + sample_offset;

        // Only used in the non-ReSTIR path
        const float sample_depth = safeTexelFetch(half_depth_tex, ivec2(sample_px), 0).r;

        // Only used in the non-ReSTIR path
        const vec4 packed1 = safeTexelFetch(hit1_tex, ivec2(sample_px), 0);

        const bool is_valid_sample = USE_RESTIR || (packed1.w > 0 && sample_depth != 0);

        if (is_valid_sample) {
            float rejection_bias = 1;

            const vec2 sample_uv = get_uv(
                sample_px * 2 + HALFRES_SUBSAMPLE_OFFSET,
                push.output_tex_size);

            // const vec4 sample_gbuffer_packed = gbuffer_tex[sample_px * 2 + HALFRES_SUBSAMPLE_OFFSET];
            // GbufferData sample_gbuffer = GbufferDataPacked::from_uint4(asuint(sample_gbuffer_packed)).unpack();

            vec3 sample_hit_normal_vs = vec3(0); // only valid without ReSTIR
            if (!USE_RESTIR) {
                sample_hit_normal_vs = safeTexelFetch(hit2_tex, ivec2(sample_px), 0).xyz;
            }

#if CUT_CORNERS_IN_MATH
            const vec3 sample_normal_vs = safeTexelFetch(half_view_normal_tex, ivec2(sample_px), 0).xyz;
#else
            const vec3 sample_normal_vs = normalize(safeTexelFetch(half_view_normal_tex, ivec2(sample_px), 0).xyz);
#endif

            vec3 sample_hit_normal_ws;

            vec3 sample_origin_ws; // only valid with ReSTIR
            vec3 sample_origin_vs;
            vec3 sample_radiance;
            float sample_ray_pdf = 1;
            float neighbor_sampling_pdf;
            vec3 center_to_hit_vs;
            vec3 center_to_hit_ws; // only valid with ReSTIR
            float sample_cos_theta;
            vec3 sample_hit_vs;

            // The bent BRDF we're using for smooth surfaces needs NdotL adjustment.
            // Rays hits close to surfaces will undergo a lot of distortion; because the bent BRDF
            // intentionally ignores parallax, this distortion can result in over-amplifying close features,
            // e.g. stretching dark lines where surfaces meet. Now, while the whole BRDF is supposed
            // to be bent, the NdotL term can be corrected. Note that this needs to be applied
            // after the PDF max value clamp, as the NdotL is not part of the PDF since we're using
            // the projected solid angle metric for BRDFs.
            float bent_pdf_ndotl_fix = 1;

            // Mult for the bent pdf
            float pdf0_mult = 1;

            // Mult for the real pdf
            float pdf1_mult = 1;

            if (USE_RESTIR) {
                uvec2 rpx = sample_px;
                const uvec2 reservoir_raw = safeTexelFetchU(restir_reservoir_tex, ivec2(rpx), 0).xy;
                Reservoir1spp r = Reservoir1spp_from_raw(reservoir_raw);
                const uvec2 spx = reservoir_payload_to_px(r.payload);

                RtrRestirRayOrigin sample_origin = RtrRestirRayOrigin_from_raw(safeTexelFetch(restir_ray_orig_tex, ivec2(spx), 0));

                sample_origin_ws = sample_origin.ray_origin_eye_offset_ws + get_eye_position(gpu_input);
                const float sample_roughness = sample_origin.roughness;

                if (
                    // Reject invalid, e.g. on sky.
                    reservoir_raw.x == 0
                    // Reject samples with much lower roughness
                    // TODO: find why this is necessar; without it, smooth surfaces surrounded by
                    // rough surfaces can get darkened. Looks like the bent lobe PDF calculation might be at fault.
                    || sample_roughness > gbuffer.roughness * 2) {
                    continue;
                }

                const vec3 sample_hit_ws = safeTexelFetch(restir_ray_tex, ivec2(spx), 0).xyz + sample_origin_ws;
                sample_origin_vs = position_world_to_view(gpu_input, sample_origin_ws);

                sample_hit_normal_ws = decode_restir_hit_normal(safeTexelFetch(restir_hit_normal_tex, ivec2(spx), 0).xyz);

                const vec3 sample_offset = sample_hit_ws - sample_origin_ws;
                const float sample_dist = length(sample_offset);
                const vec3 sample_dir = sample_offset / sample_dist;

                sample_radiance = safeTexelFetch(restir_irradiance_tex, ivec2(spx), 0).rgb;
                sample_ray_pdf = safeTexelFetch(restir_ray_tex, ivec2(spx), 0).a;
                neighbor_sampling_pdf = 1.0 / r.W;
                center_to_hit_vs = position_world_to_view(gpu_input, sample_hit_ws) - mix(refl_ray_origin_vs, sample_origin_vs, RTR_NEIGHBOR_RAY_ORIGIN_CENTER_BIAS);
                center_to_hit_ws = sample_hit_ws - mix(refl_ray_origin_ws, sample_origin_ws, RTR_NEIGHBOR_RAY_ORIGIN_CENTER_BIAS);
                sample_hit_vs = center_to_hit_vs + position_world_to_view(gpu_input, sample_origin_ws);
                sample_cos_theta = rtr_decode_cos_theta_from_fp16(safeTexelFetch(restir_irradiance_tex, ivec2(spx), 0).a);

                // Perform measure conversion

                const float center_to_hit_dist = length(center_to_hit_vs);
                const float sample_to_hit_dist = length(sample_hit_ws - sample_origin_ws);

                if (RTR_USE_BULLSHIT_TO_FIX_EDGE_HALOS) {
                    // Looks almost correct without this entire term, but then at certain roughness levels,
                    // there's a bit of a halo followed by a darker line in corners.
                    //
                    // My math is likely (definitely) nonsense somewhere, and then this other nonsense
                    // is "needed" to cancel out.
                    //
                    // Everything is fine if `RTR_NEIGHBOR_RAY_ORIGIN_CENTER_BIAS` is 1.0,
                    // and none of this nonsense is needed then, but that has its own issues.

                    const float center_to_hit_dist_wat_i_dont_even = length(
                        position_world_to_view(gpu_input, sample_hit_ws)
                        // At low roughness pretend we're directly using neighbor positions for hits ¯\_(ツ)_/¯
                        - mix(refl_ray_origin_vs, sample_origin_vs, mix(1.0, RTR_NEIGHBOR_RAY_ORIGIN_CENTER_BIAS,
                                                                        // eyeballed bullshit
                                                                        0.4 * min(1, 3 * sqrt(gbuffer.roughness)))));
                    pdf0_mult *= max(1e-5, pow(center_to_hit_dist_wat_i_dont_even / sample_to_hit_dist, 2));
                    pdf1_mult *= max(1.0, pow(center_to_hit_dist / sample_to_hit_dist, 2));
                } else {
                    // TODO: should not be clamped, but clamping reduces some excessive hotness
                    // in corners on smooth surfaces, which is a good trade of the slight darkening this causes.
                    neighbor_sampling_pdf *= max(
                        select(RTR_MEASURE_CONVERSION_CLAMP_ATTENUATION, 1.0, 1e-5),
                        pow(center_to_hit_dist / sample_to_hit_dist, 2));
                }

#if !CUT_CORNERS_IN_MATH
                if (USE_RESTIR) {
                    float center_to_hit_vis = dot(sample_hit_normal_ws, -normalize(center_to_hit_ws));
                    float sample_to_hit_vis = dot(sample_hit_normal_ws, -normalize(sample_hit_ws - sample_origin_ws));
                    neighbor_sampling_pdf /= clamp(center_to_hit_vis / sample_to_hit_vis, 1e-1, 1e1);
                } else {
                    float center_to_hit_vis = dot(sample_hit_normal_vs, -normalize(center_to_hit_vs));
                    float sample_to_hit_vis = dot(sample_hit_normal_vs, -normalize(sample_hit_vs - sample_origin_vs));
                    neighbor_sampling_pdf /= clamp(center_to_hit_vis / sample_to_hit_vis, 1e-1, 1e1);
                }
#endif

#if !RTR_APPROX_MEASURE_CONVERSION
                vec3 wi = normalize(direction_view_to_world(gpu_input, center_to_hit_vs) * tangent_to_world);

                const vec3 sample_origin_to_hit_vs = position_world_to_view(gpu_input, sample_hit_ws) - sample_origin_vs;
                const float sample_wi_z = dot(sample_normal_vs, normalize(sample_origin_to_hit_vs));
                const float wi_measure_fix = sample_wi_z / wi.z;

                neighbor_sampling_pdf *= clamp(wi_measure_fix, 1e-2, 1e2);

                // Same factor will be applied to cancel this measure conversion for the bent BRDF.
                // Since the bent BRDF doesn't undergo parallax, it should not be converting
                // the wi measure.
                bent_pdf_ndotl_fix *= clamp(wi_measure_fix, 1e-2, 1e2);
#endif
            } else {
                // The pure ratio estimator (non-ReSTIR) path.

                const vec4 packed0 = safeTexelFetch(hit0_tex, ivec2(sample_px), 0);

                const ViewRayContext sample_ray_ctx = vrc_from_uv_and_biased_depth(gpu_input, sample_uv, sample_depth);
                sample_origin_vs = ray_hit_vs(sample_ray_ctx);

                neighbor_sampling_pdf = packed1.w;
                sample_radiance = packed0.xyz;
                sample_cos_theta = 1;

                const vec3 real_sample_hit_vs = direction_world_to_view(gpu_input, packed1.xyz) + sample_origin_vs;

                center_to_hit_vs = real_sample_hit_vs - mix(refl_ray_origin_vs, sample_origin_vs, RTR_NEIGHBOR_RAY_ORIGIN_CENTER_BIAS);
                sample_hit_vs = center_to_hit_vs + refl_ray_origin_vs;
            }

            const vec3 wi = normalize(direction_view_to_world(gpu_input, center_to_hit_vs) * tangent_to_world);

            if (USE_RESTIR) {
                if (wi.z < 1e-5
                // Discard hit samples which face away from the center pixel
#if !CUT_CORNERS_IN_MATH
                    || dot(sample_hit_normal_ws, -center_to_hit_ws) <= 0
#endif
                ) {
                    continue;
                }
            } else {
                if (wi.z < 1e-5
                    // Discard hit samples which face away from the center pixel
                    || dot(sample_hit_normal_vs, -center_to_hit_vs) <= 0) {
                    continue;
                }
            }

#if 1
// Soft directional falloff
#if !CUT_CORNERS_IN_MATH
            rejection_bias *= max(1e-2, SpecularBrdf_ggx_ndf_0_1(a2, dot(normal_vs, sample_normal_vs)));
#endif

            // Hard cutoff below a pretty wide angle, to prevent contributions from
            // widely different samples than the center.
            rejection_bias *= float(dot(normal_vs, sample_normal_vs) > 0.7);

            // Depth-based rejection
            {
                const float depth_diff = abs(refl_ray_origin_vs.z - sample_origin_vs.z) / max(1e-10, kernel_size_ws);
                rejection_bias *= exp2(-max(0.3, normal_vs.z) * depth_diff * depth_diff);
            }
#endif

            float center_to_hit_dist2 = dot(center_to_hit_vs, center_to_hit_vs);

// Compensate for change in visibility term. Automatic if storing with the surface area metric.
#if !RTR_PDF_STORED_WITH_SURFACE_AREA_METRIC
            // If ReSTIR is used, measure conversion is done when loading sample data
            if (!USE_RESTIR) {
                const vec3 sample_to_hit_vs = sample_hit_vs - sample_origin_vs;
                float sample_to_hit_dist2 = dot(sample_to_hit_vs, sample_to_hit_vs);

                if (dot(sample_normal_vs, sample_to_hit_vs) <= 0) {
                    // TODO: should this run with ReSTIR too?
                    continue;
                }

                float center_to_hit_vis = dot(sample_hit_normal_vs, -normalize(center_to_hit_vs));
                float sample_to_hit_vis = dot(sample_hit_normal_vs, -normalize(sample_to_hit_vs));
                if (sample_to_hit_vis <= 1e-5) {
                    // TODO: should this run with ReSTIR too?
                    continue;
                }

                neighbor_sampling_pdf *= max(1.0, center_to_hit_dist2 / sample_to_hit_dist2);
                const float sample_wo_measure_fix = sample_to_hit_vis / center_to_hit_vis;
                const float wi_measure_fix = dot(sample_normal_vs, normalize(sample_to_hit_vs)) / wi.z;

                bent_pdf_ndotl_fix *= clamp(wi_measure_fix, 1e-2, 1e2);

#if !RTR_APPROX_MEASURE_CONVERSION
                neighbor_sampling_pdf *= clamp(sample_wo_measure_fix * wi_measure_fix, 1e-1, 1e1);
#else
                neighbor_sampling_pdf *= clamp(sample_wo_measure_fix, 1e-1, 1e1);
#endif
            }
#endif

            const vec3 surface_offset = sample_origin_vs - refl_ray_origin_vs;
#if USE_APPROXIMATE_SAMPLE_SHADOWING
            if (dot(center_to_hit_vs, normal_vs) * 0.2 / length(center_to_hit_vs) < dot(surface_offset, normal_vs) / length(surface_offset)) {
                rejection_bias *= select(is_center_sample, 1, 0);
            }
#endif

            BrdfValue spec = evaluate(specular_brdf, wo, wi);

            // The FG weight is included in the radiance accumulator,
            // so should not be in the ratio estimator weight.
            float spec_weight = spec.pdf * step(0.0, wi.z);

            float to_psa_metric = 1;
#if RTR_PDF_STORED_WITH_SURFACE_AREA_METRIC
            // Convert to projected solid angle
            to_psa_metric =
#if RTR_APPROX_MEASURE_CONVERSION
                1
#else
                abs(wi.z * dot(sample_hit_normal_vs, -normalize(center_to_hit_vs)))
#endif
                / center_to_hit_dist2;

            neighbor_sampling_pdf /= to_psa_metric;
#endif

            float contrib_wt = 0;
            if (!USE_RESTIR) {
                contrib_wt = rejection_bias * spec_weight / neighbor_sampling_pdf;
                contrib_accum += vec4(sample_radiance * spec.value_over_pdf, 1) * contrib_wt;
            } else {
                const float cos_theta = normalize(wo + wi).z;

                // Only allow the theta to be increased by a certain amont from what it should be.
                // This prevents the specular highlight from growing too much in size,
                // especially on tiny geometric features that barely catch the light.
                const float bent_cos_theta = min(sample_cos_theta, cos_theta * 1.25);

                const float sample_ray_ndf = SpecularBrdf_ggx_ndf(a2, bent_cos_theta);
                const float center_ndf = SpecularBrdf_ggx_ndf(a2, cos_theta);

                float bent_sample_pdf = spec.pdf * sample_ray_ndf / center_ndf;

                // Blent towards using the real spec pdf at high roughness. Otherwise the ratio
                // estimator combined with bent sample rays results in too much brightness in corners/cracks.
                // At low roughness this can result in noise in the FG estimate, causing darkening.
                const float pdf_lerp_t = smoothstep(0.4, 0.7, sqrt(gbuffer.roughness)) * smoothstep(0.0, 0.1, ray_len_avg / eye_to_surf_dist);

                const vec3 pdfs[2] = {
                    vec3(
                        min(bent_sample_pdf, RTR_RESTIR_MAX_PDF_CLAMP) * bent_pdf_ndotl_fix,
                        neighbor_sampling_pdf * pdf0_mult,
                        1 - pdf_lerp_t),
                    vec3(
                        min(spec.pdf, RTR_RESTIR_MAX_PDF_CLAMP),
                        neighbor_sampling_pdf * pdf1_mult,
                        pdf_lerp_t),
                };

                for (uint pdf_i = 0; pdf_i < 2; ++pdf_i) {
                    const float bent_sample_pdf = pdfs[pdf_i].x;
                    const float neighbor_sampling_pdf = pdfs[pdf_i].y;
                    const float pdf_influence = pdfs[pdf_i].z;

                    // Pseudo MIS. Weigh down samples which claim to be high pdf
                    // if we could have sampled them with a lower pdf. This helps rough reflections
                    // surrounded by almost-mirrors. The rays sampled from the mirrors will have
                    // high pdf values, and skew the integration, creating halos around themselves
                    // on the rough objects.
                    // This is similar in formulation to actual MIS; where we're lying is in claiming
                    // that the central pixel generates samples, while it does not.
                    //
                    // The error from this seems to be making detail in roughness map hotter,
                    // which is not a terrible thing given that it's also usually dimmed down
                    // by temporal filters.
                    float mis_weight = max(1e-4, spec.pdf / (sample_ray_pdf + spec.pdf));

                    contrib_wt = rejection_bias * mis_weight * max(1e-10, spec_weight / bent_sample_pdf);

                    if (pdf_i == 0) {
                        safeImageStore(rtr_debug_image, ivec2(px), vec4(vec3(bent_sample_pdf / neighbor_sampling_pdf * spec.value_over_pdf * contrib_wt * pdf_influence), 1));
                    }

                    contrib_accum += vec4(
                                         sample_radiance * bent_sample_pdf / neighbor_sampling_pdf * spec.value_over_pdf,
                                         1) *
                                     contrib_wt * pdf_influence;
                }
            }

            // float lum = sRGB_to_luminance(sample_radiance);
            // ex += lum * contrib_wt;
            // ex2 += lum * lum * contrib_wt;

            // Aggressively bias towards closer hits
            ray_len_accum += exponential_squish(surf_to_hit_dist, ray_squish_scale) * contrib_wt;
            sample_radius_accum += 1.0 - RADIUS_INC_ON_FAIL;
        }
    }

    // Note: When the borrow sample count is low (or 1), normalizing here
    // introduces a heavy bias, as it disregards the PDFs with which
    // samples have been taken.

    // Note: do not include the clamp range from `rejection_bias`
    const float contrib_norm_factor = max(1e-14, contrib_accum.w);

    contrib_accum.rgb /= contrib_norm_factor;
    // ex /= contrib_norm_factor;
    // ex2 /= contrib_norm_factor;
    ray_len_accum /= contrib_norm_factor;

    SpecularBrdfEnergyPreservation brdf_lut = SpecularBrdfEnergyPreservation_from_brdf_ndotv(specular_brdf, wo.z);

#if !RTR_RENDER_SCALED_BY_FG
    contrib_accum.rgb /= brdf_lut.preintegrated_reflection;
#endif

    // The invalid samples are in reality multi-scater events, and here we adjust for that.
    // Note that while the `valid_sample_fraction` is a grayscale multiplier,
    // this is chromatic, and will cause an increase in saturation on conductors.
    contrib_accum.rgb *= brdf_lut.preintegrated_reflection_mult;

    ray_len_accum = exponential_unsquish(ray_len_accum, ray_squish_scale);

    vec3 out_color = contrib_accum.rgb;

    // float relative_error = sqrt(max(0.0, ex2 - ex * ex)) / max(1e-5, ex2);
    safeImageStore(output_tex, ivec2(px), vec4(out_color, 0));
    safeImageStore(ray_len_output_tex, ivec2(px), vec4(ray_len_accum, ray_len_avg, 0, 0));
}

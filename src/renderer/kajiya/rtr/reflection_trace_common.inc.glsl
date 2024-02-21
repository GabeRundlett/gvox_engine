// Large enough to mean "far away" and small enough so that
// the hit points/vectors fit within fp16.
const float SKY_DIST = 1e4;

const bool USE_SOFT_SHADOWS = true;
const bool USE_SOFT_SHADOWS_TEMPORAL_JITTER = false;

const bool USE_TEMPORAL_JITTER = true;

// Should be off then iterating on reflections,
// but might be a good idea to enable for shipping anything.
#define USE_HEAVY_BIAS 1

#define USE_WORLD_RADIANCE_CACHE 0
#define LOWEST_ROUGHNESS_FOR_RADIANCE_CACHE 0.5

const bool USE_IRCACHE = true;

// Note: should be off when using dedicated specular lighting passes in addition to RTR
const bool USE_EMISSIVE = true;
const bool USE_LIGHTS = true;

// Debug bias in sample reuse with position-based hit storage
#define COLOR_CODE_GROUND_SKY_BLACK_WHITE 0

// Strongly reduces roughness of secondary hits
#define USE_AGGRESSIVE_SECONDARY_ROUGHNESS_BIAS 1

// BRDF bias
#define SAMPLING_BIAS 0.05

const bool USE_SCREEN_GI_REPROJECTION = true;

#if USE_HEAVY_BIAS
#undef USE_AGGRESSIVE_SECONDARY_ROUGHNESS_BIAS
#define USE_AGGRESSIVE_SECONDARY_ROUGHNESS_BIAS 1

#undef SAMPLING_BIAS
#define SAMPLING_BIAS 0.15
#endif

#include <renderer/atmosphere/sky.glsl>
#include "../inc/ray_cone.glsl"
#include "../inc/safety.glsl"

struct RtrTraceResult {
    vec3 total_radiance;
    float hit_t;
    vec3 hit_normal_vs;
};

bool rt_is_shadowed(RayDesc ray) {
    ShadowRayPayload shadow_payload = ShadowRayPayload_new_hit();
    VoxelTraceResult trace_result = voxel_trace(VoxelTraceInfo(VOXELS_BUFFER_PTRS, ray.Direction, MAX_STEPS, ray.TMax, ray.TMin, true), ray.Origin);
    shadow_payload.is_shadowed = trace_result.dist < ray.TMax;
    return shadow_payload.is_shadowed;
}

RtrTraceResult do_the_thing(uvec2 px, vec3 normal_ws, float roughness, inout uint rng, RayDesc outgoing_ray) {
#if USE_AGGRESSIVE_SECONDARY_ROUGHNESS_BIAS
    const float roughness_bias = roughness;
#else
    const float roughness_bias = 0.5 * roughness;
#endif

    // See note in `assets/shaders/rtr/resolve.hlsl`
    const float reflected_cone_spread_angle = sqrt(roughness) * 0.05;

    const RayCone ray_cone = propagate(
        pixel_ray_cone_from_image_height(gpu_input, push.gbuffer_tex_size.y * 0.5),
        reflected_cone_spread_angle, length(outgoing_ray.Origin - get_eye_position(gpu_input)));

    if (LAYERED_BRDF_FORCE_DIFFUSE_ONLY == 0) {
        GbufferRaytrace primary_hit_ = GbufferRaytrace_with_ray(outgoing_ray);
        primary_hit_ = with_cone(primary_hit_, ray_cone);
        primary_hit_ = with_cull_back_faces(primary_hit_, false);
        primary_hit_ = with_path_length(primary_hit_, 1);
        const GbufferPathVertex primary_hit = trace(primary_hit_, VOXELS_BUFFER_PTRS);

        if (primary_hit.is_hit) {
            GbufferData gbuffer = unpack(primary_hit.gbuffer_packed);
            gbuffer.roughness = mix(gbuffer.roughness, 1.0, roughness_bias);
            const mat3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);
            const vec3 wo = (-outgoing_ray.Direction) * tangent_to_world;
            const LayeredBrdf brdf = LayeredBrdf_from_gbuffer_ndotv(gbuffer, wo.z);

            // Project the sample into clip space, and check if it's on-screen
            const vec3 primary_hit_cs = position_world_to_sample(gpu_input, primary_hit.position);
            const vec2 primary_hit_uv = cs_to_uv(primary_hit_cs.xy);
            const float primary_hit_screen_depth = textureLod(daxa_sampler2D(depth_tex, g_sampler_nnc), primary_hit_uv, 0).r;
            GbufferData primary_hit_screen_gbuffer = unpack(GbufferDataPacked(safeTexelFetchU(gbuffer_tex, ivec2(primary_hit_uv * push.gbuffer_tex_size.xy), 0)));
            const bool is_on_screen =
                all(lessThan(abs(primary_hit_cs.xy), vec2(1.0))) &&
                inverse_depth_relative_diff(primary_hit_cs.z, primary_hit_screen_depth) < 5e-3 &&
                dot(primary_hit_screen_gbuffer.normal, -outgoing_ray.Direction) > 0.0 &&
                dot(primary_hit_screen_gbuffer.normal, gbuffer.normal) > 0.7;

            vec3 total_radiance = 0.0.xxx;
            vec3 reflected_normal_vs;
            {
                // Sun
                vec3 sun_radiance = sun_radiance_in_direction(gpu_input, transmittance_lut, SUN_DIRECTION);
                {
#if 1
                    const vec2 urand = vec2(
                        uint_to_u01_float(hash1_mut(rng)),
                        uint_to_u01_float(hash1_mut(rng)));
#else
                    const vec2 urand = blue_noise_for_pixel(
                                           px,
                                           select(USE_SOFT_SHADOWS_TEMPORAL_JITTER, frame_constants.frame_index, 0))
                                           .xy;
#endif

                    const vec3 to_light_norm = sample_sun_direction(
                        gpu_input,
                        urand,
                        USE_SOFT_SHADOWS);

                    const bool is_shadowed = rt_is_shadowed(new_ray(
                        primary_hit.position,
                        to_light_norm,
                        1e-4,
                        SKY_DIST));

                    const vec3 wi = to_light_norm * tangent_to_world;

                    const vec3 brdf_value = evaluate(brdf, wo, wi) * max(0.0, wi.z);
                    const vec3 light_radiance = select(bvec3(is_shadowed), vec3(0.0), sun_radiance);
                    total_radiance += brdf_value * light_radiance;
                }

                reflected_normal_vs = direction_world_to_view(gpu_input, gbuffer.normal);

                if (USE_EMISSIVE) {
                    total_radiance += gbuffer.emissive;
                }

                if (USE_SCREEN_GI_REPROJECTION && is_on_screen) {
                    const vec3 reprojected_radiance =
                        textureLod(daxa_sampler2D(rtdgi_tex, g_sampler_nnc), primary_hit_uv, 0).rgb * deref(gpu_input).pre_exposure_delta;

                    total_radiance += reprojected_radiance.rgb * gbuffer.albedo;
                } else {
                    // if (USE_LIGHTS) {
                    //     vec2 urand = vec2(
                    //         uint_to_u01_float(hash1_mut(rng)),
                    //         uint_to_u01_float(hash1_mut(rng)));

                    //     for (uint light_idx = 0; light_idx < frame_constants.triangle_light_count; light_idx += 1) {
                    //         TriangleLight triangle_light = TriangleLight::from_packed(triangle_lights_dyn[light_idx]);
                    //         LightSampleResultArea light_sample = sample_triangle_light(triangle_light.as_triangle(), urand);
                    //         const vec3 shadow_ray_origin = primary_hit.position;
                    //         const vec3 to_light_ws = light_sample.pos - shadow_ray_origin;
                    //         const float dist_to_light2 = dot(to_light_ws, to_light_ws);
                    //         const vec3 to_light_norm_ws = to_light_ws * rsqrt(dist_to_light2);

                    //         const float to_psa_metric =
                    //             max(0.0, dot(to_light_norm_ws, gbuffer.normal)) * max(0.0, dot(to_light_norm_ws, -light_sample.normal)) / dist_to_light2;

                    //         if (to_psa_metric > 0.0) {
                    //             const bool is_shadowed =
                    //                 rt_is_shadowed(
                    //                     acceleration_structure,
                    //                     new_ray(
                    //                         shadow_ray_origin,
                    //                         to_light_norm_ws,
                    //                         1e-4,
                    //                         sqrt(dist_to_light2) - 2e-4));

                    //             #if 1
                    //                 const vec3 bounce_albedo = max(gbuffer.albedo, 1.0.xxx, 0.04);
                    //                 const vec3 brdf_value = bounce_albedo * to_psa_metric / M_PI;
                    //             #else
                    //                 const vec3 wi = mul(to_light_norm_ws, tangent_to_world);
                    //                 const vec3 brdf_value = brdf.evaluate(wo, wi) * to_psa_metric;
                    //             #endif

                    //             total_radiance +=
                    //                 select(!is_shadowed, (triangle_light.radiance() * brdf_value / light_sample.pdf.value), 0);
                    //         }
                    //     }
                    // }

                    if (USE_IRCACHE) {
                        const float cone_width = propagate(ray_cone, 0, primary_hit.ray_t).width;

                        IrcacheLookupParams new_params = IrcacheLookupParams_create(
                            outgoing_ray.Origin, primary_hit.position, gbuffer.normal);
                        new_params = with_query_rank(new_params, 1);
                        new_params = with_stochastic_interpolation(new_params, cone_width < 0.1);
                        const vec3 gi = lookup(new_params, rng);

                        total_radiance += gi * gbuffer.albedo;
                    }
                }
            }

            RtrTraceResult result;

#if COLOR_CODE_GROUND_SKY_BLACK_WHITE
            result.total_radiance = 0.0.xxx;
#else
            result.total_radiance = total_radiance;
#endif

            result.hit_t = primary_hit.ray_t;
            result.hit_normal_vs = reflected_normal_vs;

            return result;
        }
    }

    RtrTraceResult result;

    float hit_t = SKY_DIST;
    vec3 far_gi;
    far_gi = sky_radiance_in_direction(gpu_input, sky_lut, transmittance_lut, outgoing_ray.Direction);

#if COLOR_CODE_GROUND_SKY_BLACK_WHITE
    result.total_radiance = 2.0.xxx;
#else
    result.total_radiance = far_gi;
#endif

    result.hit_t = hit_t;
    result.hit_normal_vs = -direction_world_to_view(gpu_input, outgoing_ray.Direction);

    return result;
}

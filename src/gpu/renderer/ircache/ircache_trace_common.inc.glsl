#pragma once
// HACK: reduces feedback loops due to the spherical traces.
// As a side effect, dims down the result a bit, and increases variance.
// Maybe not needed when using IRCACHE_LOOKUP_PRECISE.
#define USE_SELF_LIGHTING_LIMITER 1

#define USE_WORLD_RADIANCE_CACHE 0

#define USE_BLEND_RESULT 0

#define SUN_COLOR vec3(1)
#include <utils/sky.glsl>

#define VOXEL_TRACE_WORLDSPACE 1
#include <voxels/core.glsl>

bool rt_is_shadowed(RayDesc ray) {
    ShadowRayPayload shadow_payload = ShadowRayPayload_new_hit();
    VoxelTraceResult trace_result = voxel_trace(VoxelTraceInfo(VOXELS_BUFFER_PTRS, ray.Direction, MAX_STEPS, ray.TMax, ray.TMin, true), ray.Origin);
    shadow_payload.is_shadowed = trace_result.dist < ray.TMax;
    return shadow_payload.is_shadowed;
}

// Rough-smooth-rough specular paths are a major source of fireflies.
// Enabling this option will bias roughness of path vertices following
// reflections off rough interfaces.
const bool FIREFLY_SUPPRESSION = true;
const bool USE_LIGHTS = true;
const bool USE_EMISSIVE = true;
const bool SAMPLE_IRCACHE_AT_LAST_VERTEX = true;
const uint MAX_PATH_LENGTH = 1;

vec3 sample_environment_light(vec3 dir) {
    return texture(daxa_samplerCube(sky_cube_tex, deref(gpu_input).sampler_llr), dir).rgb;
}

float pack_dist(float x) {
    return min(1, x);
}

float unpack_dist(float x) {
    return x;
}

struct IrcacheTraceResult {
    vec3 incident_radiance;
    vec3 direction;
    vec3 hit_pos;
};

IrcacheTraceResult ircache_trace(Vertex entry, DiffuseBrdf brdf, SampleParams sample_params, uint life) {
    const mat3 tangent_to_world = build_orthonormal_basis(entry.normal);

    uint rng = rng(sample_params);

    RayDesc outgoing_ray = new_ray(
        entry.position,
        direction(sample_params),
        0.0,
        FLT_MAX);

    // force rays in the direction of the normal (debug)
    // outgoing_ray.Direction = mul(tangent_to_world, vec3(0, 0, 1));

    IrcacheTraceResult result;
    result.direction = outgoing_ray.Direction;

    // WrcFarField far_field = WrcFarField::create_miss();
    // if (far_field.is_hit()) {
    //     outgoing_ray.TMax = far_field.probe_t;
    // }

    // ----

    vec3 throughput = 1.0.xxx;
    float roughness_bias = 0.5;

    vec3 irradiance_sum = vec3(0);
    vec2 hit_dist_wt = vec2(0);

    for (uint path_length = 0; path_length < MAX_PATH_LENGTH; ++path_length) {
        GbufferRaytrace gbuffer_raytrace_ = GbufferRaytrace_with_ray(outgoing_ray);
        gbuffer_raytrace_ = with_cone(gbuffer_raytrace_, RayCone_from_spread_angle(0.1));
        gbuffer_raytrace_ = with_cull_back_faces(gbuffer_raytrace_, false);
        gbuffer_raytrace_ = with_path_length(gbuffer_raytrace_, path_length + 1); // +1 because this is indirect light
        gbuffer_raytrace_ = with_cull_back_faces(gbuffer_raytrace_, false);
        const GbufferPathVertex primary_hit = trace(gbuffer_raytrace_, VOXELS_BUFFER_PTRS);

        if (primary_hit.is_hit) {
            if (0 == path_length) {
                result.hit_pos = primary_hit.position;
            }

            const vec3 to_light_norm = SUN_DIRECTION;

            const bool is_shadowed = rt_is_shadowed(new_ray(
                primary_hit.position,
                to_light_norm,
                1e-4,
                FLT_MAX));

            if (0 == path_length) {
                hit_dist_wt += vec2(pack_dist(primary_hit.ray_t), 1);
            }

            GbufferData gbuffer = unpack(primary_hit.gbuffer_packed);

            const mat3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);
            const vec3 wi = to_light_norm * tangent_to_world;

            vec3 wo = (-outgoing_ray.Direction) * tangent_to_world;

            // Hack for shading normals facing away from the outgoing ray's direction:
            // We flip the outgoing ray along the shading normal, so that the reflection's curvature
            // continues, albeit at a lower rate.
            if (wo.z < 0.0) {
                wo.z *= -0.25;
                wo = normalize(wo);
            }

            LayeredBrdf brdf = LayeredBrdf_from_gbuffer_ndotv(gbuffer, wo.z);

            if (FIREFLY_SUPPRESSION) {
                brdf.specular_brdf.roughness = mix(brdf.specular_brdf.roughness, 1.0, roughness_bias);
            }

            const vec3 brdf_value = evaluate_directional_light(brdf, wo, wi);
            const vec3 light_radiance = select(bvec3(is_shadowed), vec3(0.0), SUN_COLOR);
            irradiance_sum += throughput * brdf_value * light_radiance * max(0.0, wi.z);

            if (USE_EMISSIVE) {
                irradiance_sum += gbuffer.emissive * throughput;
            }

            // if (USE_LIGHTS && frame_constants.triangle_light_count > 0 /* && path_length > 0*/) { // rtr comp
            //     const float light_selection_pmf = 1.0 / frame_constants.triangle_light_count;
            //     const uint light_idx = hash1_mut(rng) % frame_constants.triangle_light_count;
            //     // const float light_selection_pmf = 1;
            //     // for (uint light_idx = 0; light_idx < frame_constants.triangle_light_count; light_idx += 1)
            //     {
            //         const vec2 urand = vec2(
            //             uint_to_u01_float(hash1_mut(rng)),
            //             uint_to_u01_float(hash1_mut(rng)));

            //         TriangleLight triangle_light = TriangleLight::from_packed(triangle_lights_dyn[light_idx]);
            //         LightSampleResultArea light_sample = sample_triangle_light(triangle_light.as_triangle(), urand);
            //         const vec3 shadow_ray_origin = primary_hit.position;
            //         const vec3 to_light_ws = light_sample.pos - primary_hit.position;
            //         const float dist_to_light2 = dot(to_light_ws, to_light_ws);
            //         const vec3 to_light_norm_ws = to_light_ws * rsqrt(dist_to_light2);

            //         const float to_psa_metric =
            //             max(0.0, dot(to_light_norm_ws, gbuffer.normal)) * max(0.0, dot(to_light_norm_ws, -light_sample.normal)) / dist_to_light2;

            //         if (to_psa_metric > 0.0) {
            //             vec3 wi = to_light_norm_ws * tangent_to_world;

            //             const bool is_shadowed =
            //                 rt_is_shadowed(new_ray(
            //                     shadow_ray_origin,
            //                     to_light_norm_ws,
            //                     1e-3,
            //                     sqrt(dist_to_light2) - 2e-3));

            //             irradiance_sum +=
            //                 select(is_shadowed, 0,
            //                        throughput * triangle_light.radiance() * brdf.evaluate(wo, wi) / light_sample.pdf.value * to_psa_metric / light_selection_pmf);
            //         }
            //     }
            // }

            if (SAMPLE_IRCACHE_AT_LAST_VERTEX && path_length + 1 == MAX_PATH_LENGTH) {
                IrcacheLookupParams new_params = IrcacheLookupParams_create(entry.position, primary_hit.position, gbuffer.normal);
                new_params = with_query_rank(new_params, 1 + ircache_entry_life_to_rank(life));
                irradiance_sum += lookup(new_params, rng) * throughput * gbuffer.albedo;
            }

            const vec3 urand = vec3(
                uint_to_u01_float(hash1_mut(rng)),
                uint_to_u01_float(hash1_mut(rng)),
                uint_to_u01_float(hash1_mut(rng)));

            BrdfSample brdf_sample = sample_brdf(brdf, wo, urand);

            // TODO: investigate NaNs here.
            if (is_valid(brdf_sample) && brdf_sample.value_over_pdf.x == brdf_sample.value_over_pdf.x) {
                roughness_bias = mix(roughness_bias, 1.0, 0.5 * brdf_sample.approx_roughness);
                outgoing_ray.Origin = primary_hit.position;
                outgoing_ray.Direction = tangent_to_world * brdf_sample.wi;
                outgoing_ray.TMin = 1e-4;
                throughput *= brdf_sample.value_over_pdf;
            } else {
                break;
            }
        } else {
            if (0 == path_length) {
                result.hit_pos = outgoing_ray.Origin + outgoing_ray.Direction * 1000;
            }

            // if (far_field.is_hit()) {
            //     irradiance_sum += throughput * far_field.radiance * far_field.inv_pdf;
            // } else
            {
                if (0 == path_length) {
                    hit_dist_wt += vec2(pack_dist(1), 1);
                }

                irradiance_sum += throughput * sample_environment_light(outgoing_ray.Direction);
            }

            break;
        }
    }

    result.incident_radiance = irradiance_sum;
    return result;
}

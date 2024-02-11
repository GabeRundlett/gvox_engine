// Should be 1, but rarely matters for the diffuse bounce, so might as well save a few cycles.
const bool USE_SOFT_SHADOWS = false;

const bool USE_IRCACHE = true;
#define USE_WORLD_RADIANCE_CACHE 0

#define ROUGHNESS_BIAS 0.5
const bool USE_SCREEN_GI_REPROJECTION = true;
#define USE_SWIZZLE_TILE_PIXELS 0

const bool USE_EMISSIVE = true;
const bool USE_LIGHTS = true;

#define USE_SKY_CUBE_TEX 1

#include <utilities/gpu/sky.glsl>

const float SKY_DIST = 1e4;

vec3 sample_environment_light(vec3 dir) {
    return texture(daxa_samplerCube(sky_cube_tex, deref(gpu_input).sampler_llr), dir).rgb;
}

uvec2 reservoir_payload_to_px(uint payload) {
    return uvec2(payload & 0xffff, payload >> 16);
}

#include <utilities/gpu/math.glsl>
#include <utilities/gpu/ray_cone.glsl>

#include <voxels/core.glsl>

bool rt_is_shadowed(RayDesc ray) {
    ShadowRayPayload shadow_payload = ShadowRayPayload_new_hit();
    VoxelTraceResult trace_result = voxel_trace(VoxelTraceInfo(VOXELS_BUFFER_PTRS, ray.Direction, MAX_STEPS, ray.TMax, ray.TMin, true), ray.Origin);
    shadow_payload.is_shadowed = trace_result.dist < ray.TMax;
    return shadow_payload.is_shadowed;
}

struct TraceResult {
    vec3 out_value;
    vec3 hit_normal_ws;
    float hit_t;
    float pdf;
    bool is_hit;
};

TraceResult do_the_thing(uvec2 px, vec3 normal_ws, inout uint rng, RayDesc outgoing_ray) {
    vec3 total_radiance = 0.0.xxx;
    vec3 hit_normal_ws = -outgoing_ray.Direction;

    float hit_t = outgoing_ray.TMax;

    // cosine-weighted
    // float pdf = 1.0 / M_PI;

    // uniform
    float pdf = max(0.0, 1.0 / (dot(normal_ws, outgoing_ray.Direction) * 2 * M_PI));

    const float reflected_cone_spread_angle = 0.03;
    const RayCone ray_cone = propagate(
        pixel_ray_cone_from_image_height(globals, push.gbuffer_tex_size.y * 0.5),
        reflected_cone_spread_angle, length(outgoing_ray.Origin - get_eye_position(globals)));

    GbufferRaytrace primary_hit_ = GbufferRaytrace_with_ray(outgoing_ray);
    primary_hit_ = with_cone(primary_hit_, ray_cone);
    primary_hit_ = with_cull_back_faces(primary_hit_, false);
    primary_hit_ = with_path_length(primary_hit_, 1);
    const GbufferPathVertex primary_hit = trace(primary_hit_, VOXELS_BUFFER_PTRS);

    if (primary_hit.is_hit) {
        hit_t = primary_hit.ray_t;
        GbufferData gbuffer = unpack(primary_hit.gbuffer_packed);
        hit_normal_ws = gbuffer.normal;

        // Project the sample into clip space, and check if it's on-screen
        const vec3 primary_hit_cs = position_world_to_sample(globals, primary_hit.position);
        const vec2 primary_hit_uv = cs_to_uv(primary_hit_cs.xy);
        const float primary_hit_screen_depth = textureLod(daxa_sampler2D(depth_tex, deref(gpu_input).sampler_nnc), primary_hit_uv, 0).r;
        // const GbufferDataPacked primary_hit_screen_gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_tex[ivec2(primary_hit_uv * gbuffer_tex_size.xy)]));
        // const vec3 primary_hit_screen_normal_ws = primary_hit_screen_gbuffer.unpack_normal();
        bool is_on_screen = true && all(lessThan(abs(primary_hit_cs.xy), vec2(1.0))) && inverse_depth_relative_diff(primary_hit_cs.z, primary_hit_screen_depth) < 5e-3
            // TODO
            //&& dot(primary_hit_screen_normal_ws, -outgoing_ray.Direction) > 0.0
            //&& dot(primary_hit_screen_normal_ws, gbuffer.normal) > 0.7
            ;

        // If it is on-screen, we'll try to use its reprojected radiance from the previous frame
        vec4 reprojected_radiance = vec4(0);
        if (is_on_screen) {
            reprojected_radiance =
                textureLod(daxa_sampler2D(reprojected_gi_tex, deref(gpu_input).sampler_nnc), primary_hit_uv, 0) * deref(gpu_input).pre_exposure_delta;

            // Check if the temporal reprojection is valid.
            is_on_screen = reprojected_radiance.w > 0;
        }

        gbuffer.roughness = mix(gbuffer.roughness, 1.0, ROUGHNESS_BIAS);
        const mat3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);
        const vec3 wo = (-outgoing_ray.Direction) * tangent_to_world;
        const LayeredBrdf brdf = LayeredBrdf_from_gbuffer_ndotv(gbuffer, wo.z);

        // Sun
        vec3 sun_radiance = sun_color_in_direction(gpu_input, transmittance_lut, SUN_DIRECTION);

        if (any(greaterThan(sun_radiance, vec3(0)))) {
            const vec3 to_light_norm = sample_sun_direction(
                gpu_input,
                blue_noise_for_pixel(blue_noise_vec2, px, rng).xy,
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

        if (USE_EMISSIVE) {
            total_radiance += gbuffer.emissive;
        }

        if (USE_SCREEN_GI_REPROJECTION && is_on_screen) {
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
            //                         1e-3,
            //                         sqrt(dist_to_light2) - 2e-3));

            //             const vec3 bounce_albedo = max(gbuffer.albedo, 1.0.xxx, 0.04);
            //             const vec3 brdf_value = bounce_albedo * to_psa_metric / M_PI;

            //             total_radiance +=
            //                 select(!is_shadowed, (triangle_light.radiance() * brdf_value / light_sample.pdf.value), 0);
            //         }
            //     }
            // }

            if (USE_IRCACHE) {
                IrcacheLookupParams new_params = IrcacheLookupParams_create(
                    outgoing_ray.Origin, primary_hit.position, gbuffer.normal);
                new_params = with_query_rank(new_params, 1);
                const vec3 gi = lookup(new_params, rng);

                total_radiance += gi * gbuffer.albedo;
            }
        }
    } else {
        // if (far_field.is_hit()) {
        //     total_radiance += far_field.radiance;
        //     hit_t = far_field.approx_surface_t;
        //     pdf = 1.0 / far_field.inv_pdf;
        // } else {
        total_radiance += sample_environment_light(outgoing_ray.Direction);
        // }
    }

    vec3 out_value = total_radiance;

    // out_value /= reservoir.p_sel;

    TraceResult result;
    result.out_value = out_value;
    result.hit_t = hit_t;
    result.hit_normal_ws = hit_normal_ws;
    result.pdf = pdf;
    result.is_hit = primary_hit.is_hit;
    return result;
}

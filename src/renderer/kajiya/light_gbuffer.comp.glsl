#include <renderer/kajiya/light_gbuffer.inl>
#include "inc/camera.glsl"

DAXA_DECL_PUSH_CONSTANT(LightGbufferComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewIndex gbuffer_tex = push.uses.gbuffer_tex;
daxa_ImageViewIndex depth_tex = push.uses.depth_tex;
daxa_ImageViewIndex shadow_mask_tex = push.uses.shadow_mask_tex;
daxa_ImageViewIndex rtr_tex = push.uses.rtr_tex;
daxa_ImageViewIndex rtdgi_tex = push.uses.rtdgi_tex;
daxa_ImageViewIndex output_tex = push.uses.output_tex;
daxa_ImageViewIndex unconvolved_sky_cube_tex = push.uses.unconvolved_sky_cube_tex;
daxa_ImageViewIndex sky_cube_tex = push.uses.sky_cube_tex;
daxa_ImageViewIndex transmittance_lut = push.uses.transmittance_lut;
// IRCACHE_USE_BUFFERS_PUSH_USES()

#include "inc/rt.glsl"
#include <renderer/sky.glsl>
#include "inc/gbuffer.glsl"

#include "inc/layered_brdf.glsl"

// #define IRCACHE_LOOKUP_DONT_KEEP_ALIVE
// #include <renderer/kajiya/ircache/lookup.glsl>

#define USE_RTDGI true
#define USE_RTR true

#define RTR_RENDER_SCALED_BY_FG false

#define USE_DIFFUSE_GI_FOR_ROUGH_SPEC false
#define USE_DIFFUSE_GI_FOR_ROUGH_SPEC_MIN_ROUGHNESS 0.7

#define FORCE_IRCACHE_DEBUG false

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    uvec2 px = gl_GlobalInvocationID.xy;
    vec2 uv = get_uv(px, push.output_tex_size);

    if (any(greaterThanEqual(px, uvec2(push.output_tex_size.xy)))) {
        return;
    }

    // if (FORCE_IRCACHE_DEBUG || push.debug_shading_mode == SHADING_MODE_IRCACHE) {
    //     if (px.y < 50) {
    //         vec3 output_ = vec3(0);
    //         const uint entry_count = deref(ircache_meta_buf).entry_count;
    //         const uint entry_alloc_count = deref(ircache_meta_buf).alloc_count;
    //         const float u = float(px.x + 0.5) * push.output_tex_size.z;
    //         const uint MAX_ENTRY_COUNT = 64 * 1024;
    //         if (px.y < 25) {
    //             if (entry_alloc_count > u * MAX_ENTRY_COUNT) {
    //                 output_ = vec3(0.05, 1, .2) * 4;
    //             }
    //         } else {
    //             if (entry_count > u * MAX_ENTRY_COUNT) {
    //                 output_ = vec3(1, 0.1, 0.05) * 4;
    //             }
    //         }
    //         // Ticks every 16k
    //         if (fract(u * 16) < push.output_tex_size.z * 32) {
    //             output_ = vec3(1, 1, 0) * 10;
    //         }
    //         imageStore(daxa_image2D(output_tex), ivec2(px), vec4(output_, 1.0));
    //         return;
    //     }
    // }

    RayDesc outgoing_ray;
    ViewRayContext view_ray_context = vrc_from_uv(globals, uv);
    {
        outgoing_ray = new_ray(
            ray_origin_ws(view_ray_context),
            ray_dir_ws(view_ray_context),
            0.0,
            FLT_MAX);
    }

    GbufferDataPacked gbuffer_packed = GbufferDataPacked(texelFetch(daxa_utexture2D(gbuffer_tex), ivec2(px), 0));
    const float depth = uintBitsToFloat(gbuffer_packed.data0.z);

    if (depth == 0.0) {
        // Render the sun disk

        // Allow the size to be changed, but don't go below the real sun's size,
        // so that we have something in the sky.
        const float real_sun_angular_radius = 0.53 * 0.5 * M_PI / 180.0;
        const float sun_angular_radius_cos = min(cos(real_sun_angular_radius), deref(gpu_input).sky_settings.sun_angular_radius_cos);

        // Conserve the sun's energy by making it dimmer as it increases in size
        // Note that specular isn't quite correct with this since we're not using area lights.
        float current_sun_angular_radius = acos(sun_angular_radius_cos);
        float sun_radius_ratio = real_sun_angular_radius / current_sun_angular_radius;

        vec3 output_ = texture(daxa_samplerCube(unconvolved_sky_cube_tex, g_sampler_llr), outgoing_ray.Direction).rgb;
        if (dot(outgoing_ray.Direction, SUN_DIRECTION) > sun_angular_radius_cos) {
            // TODO: what's the correct value?
            output_ += 800.0 * sun_color_in_direction(gpu_input, transmittance_lut, outgoing_ray.Direction) * sun_radius_ratio * sun_radius_ratio;
        }

        output_ *= deref(gpu_input).pre_exposure;
        // temporal_output_tex[px] = vec4(output_, 1);
        imageStore(daxa_image2D(output_tex), ivec2(px), vec4(output_, 1.0));
        return;
    }

    const vec3 to_light_norm = SUN_DIRECTION;

    float shadow_mask = texelFetch(daxa_texture2D(shadow_mask_tex), ivec2(px), 0).x;

    if (push.debug_shading_mode == SHADING_MODE_RTX_OFF) {
        shadow_mask = 1;
    }

    GbufferData gbuffer = unpack(gbuffer_packed);

    if (push.debug_shading_mode == SHADING_MODE_NO_TEXTURES) {
        gbuffer.albedo = vec3(0.5);
    }

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
    const vec3 brdf_value = evaluate_directional_light(brdf, wo, wi) * max(0.0, wi.z);
    const vec3 light_radiance = shadow_mask * sun_color_in_direction(gpu_input, transmittance_lut, SUN_DIRECTION);
    vec3 total_radiance = brdf_value * light_radiance;

    total_radiance += gbuffer.emissive;

    vec3 gi_irradiance = vec3(0.0);

    if (push.debug_shading_mode != SHADING_MODE_RTX_OFF) {
        if (USE_RTDGI) {
            gi_irradiance = texelFetch(daxa_texture2D(rtdgi_tex), ivec2(px), 0).rgb;
        }
    }

    // gi_irradiance += vec3(5.0);

    if (LAYERED_BRDF_FORCE_DIFFUSE_ONLY != 0) {
        total_radiance += gi_irradiance * brdf.diffuse_brdf.albedo;
    } else {
        total_radiance += gi_irradiance * brdf.diffuse_brdf.albedo * brdf.energy_preservation.preintegrated_transmission_fraction;
    }

    if (USE_RTR && LAYERED_BRDF_FORCE_DIFFUSE_ONLY == 0 && push.debug_shading_mode != SHADING_MODE_RTX_OFF) {
        vec3 rtr_radiance;

        if (!RTR_RENDER_SCALED_BY_FG) {
            rtr_radiance = texelFetch(daxa_texture2D(rtr_tex), ivec2(px), 0).xyz * brdf.energy_preservation.preintegrated_reflection;
        } else {
            rtr_radiance = texelFetch(daxa_texture2D(rtr_tex), ivec2(px), 0).xyz;
        }

        if (USE_DIFFUSE_GI_FOR_ROUGH_SPEC) {
            rtr_radiance = mix(
                rtr_radiance,
                gi_irradiance * brdf.energy_preservation.preintegrated_reflection,
                smoothstep(USE_DIFFUSE_GI_FOR_ROUGH_SPEC_MIN_ROUGHNESS, mix(USE_DIFFUSE_GI_FOR_ROUGH_SPEC_MIN_ROUGHNESS, 1.0, 0.5), gbuffer.roughness));
        }

        total_radiance += rtr_radiance;
    }

    // temporal_output_tex[px] = vec4(total_radiance, 1.0);

    vec3 output_ = total_radiance;

    // vec4 pt_cs = vec4(uv_to_cs(uv), depth, 1.0);
    // vec4 pt_ws = deref(globals).player.cam.view_to_world * deref(globals).player.cam.sample_to_view * pt_cs;
    // pt_ws /= pt_ws.w;
    // uint rng = hash3(uvec3(px, deref(gpu_input).frame_index));
    // if (FORCE_IRCACHE_DEBUG || push.debug_shading_mode == SHADING_MODE_IRCACHE) {
    //     IrcacheLookupParams ircache_params = IrcacheLookupParams_create(get_eye_position(globals), pt_ws.xyz, gbuffer.normal);
    //     output_ = lookup(ircache_params, rng);
    // }

    output_ *= deref(gpu_input).pre_exposure;

    // output_ = gbuffer.albedo;
    imageStore(daxa_image2D(output_tex), ivec2(px), vec4(output_, 1.0));
}

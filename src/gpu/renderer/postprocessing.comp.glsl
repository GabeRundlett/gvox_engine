#include <shared/app.inl>
#include <utils/math.glsl>

#if LightGbufferComputeShader

#include <utils/rt.glsl>
#include <utils/sky.glsl>
#include <utils/gbuffer.glsl>

#include <utils/layered_brdf.glsl>

// #define IRCACHE_LOOKUP_DONT_KEEP_ALIVE
#include <renderer/ircache/lookup.glsl>

#define USE_RTDGI true
#define USE_RTR true

#define RTR_RENDER_SCALED_BY_FG false

#define USE_DIFFUSE_GI_FOR_ROUGH_SPEC false
#define USE_DIFFUSE_GI_FOR_ROUGH_SPEC_MIN_ROUGHNESS 0.7

#define FORCE_IRCACHE_DEBUG true

// #include "inc/atmosphere.hlsl"
// #include "inc/sun.hlsl"

vec3 sun_color_in_direction(vec3 nrm) {
    AtmosphereLightingInfo sun_lighting = get_atmosphere_lighting(sky_lut, transmittance_lut, nrm, vec3(0, 0, 1));
    return sun_lighting.sun_direct_illuminance;
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    uvec2 px = gl_GlobalInvocationID.xy;
    vec2 uv = get_uv(px, push.output_tex_size);

    if (any(greaterThanEqual(px, uvec2(push.output_tex_size.xy)))) {
        return;
    }

    if (FORCE_IRCACHE_DEBUG || push.debug_shading_mode == SHADING_MODE_IRCACHE) {
        if (px.y < 50) {
            vec3 output_ = vec3(0);
            const uint entry_count = deref(ircache_meta_buf).entry_count;
            const uint entry_alloc_count = deref(ircache_meta_buf).alloc_count;

            const float u = float(px.x + 0.5) * push.output_tex_size.z;

            const uint MAX_ENTRY_COUNT = 64 * 1024;

            if (px.y < 25) {
                if (entry_alloc_count > u * MAX_ENTRY_COUNT) {
                    output_ = vec3(0.05, 1, .2) * 4;
                }
            } else {
                if (entry_count > u * MAX_ENTRY_COUNT) {
                    output_ = vec3(1, 0.1, 0.05) * 4;
                }
            }

            // Ticks every 16k
            if (fract(u * 16) < push.output_tex_size.z * 32) {
                output_ = vec3(1, 1, 0) * 10;
            }

            imageStore(daxa_image2D(output_tex), daxa_i32vec2(px), daxa_f32vec4(output_, 1.0));
            return;
        }
    }

    RayDesc outgoing_ray;
    ViewRayContext view_ray_context = vrc_from_uv(globals, uv);
    {
        outgoing_ray = new_ray(
            ray_origin_ws(view_ray_context),
            ray_dir_ws(view_ray_context),
            0.0,
            FLT_MAX);
    }

    GbufferDataPacked gbuffer_packed = GbufferDataPacked(texelFetch(daxa_utexture2D(gbuffer_tex), daxa_i32vec2(px), 0));
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

        vec3 output_ = texture(daxa_samplerCube(unconvolved_sky_cube_tex, deref(gpu_input).sampler_llr), outgoing_ray.Direction).rgb;
        if (dot(outgoing_ray.Direction, SUN_DIRECTION) > sun_angular_radius_cos) {
            // TODO: what's the correct value?
            output_ += 800.0 * sun_color_in_direction(outgoing_ray.Direction) * sun_radius_ratio * sun_radius_ratio;
        }

        output_ *= deref(gpu_input).pre_exposure;
        // temporal_output_tex[px] = vec4(output_, 1);
        imageStore(daxa_image2D(output_tex), daxa_i32vec2(px), daxa_f32vec4(output_, 1.0));
        return;
    }

    const vec3 to_light_norm = SUN_DIRECTION;

    float shadow_mask = texelFetch(daxa_texture2D(shadow_mask_tex), daxa_i32vec2(px), 0).x;

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
    const vec3 light_radiance = shadow_mask * sun_color_in_direction(SUN_DIRECTION);
    vec3 total_radiance = brdf_value * light_radiance;

    total_radiance += gbuffer.emissive;

    vec3 gi_irradiance = vec3(0.0);

    if (push.debug_shading_mode != SHADING_MODE_RTX_OFF) {
        if (USE_RTDGI) {
            gi_irradiance = texelFetch(daxa_texture2D(rtdgi_tex), daxa_i32vec2(px), 0).rgb;
        }
    }

    // gi_irradiance += vec3(1.0);

    if (LAYERED_BRDF_FORCE_DIFFUSE_ONLY) {
        total_radiance += gi_irradiance * brdf.diffuse_brdf.albedo;
    } else {
        total_radiance += gi_irradiance * brdf.diffuse_brdf.albedo * brdf.energy_preservation.preintegrated_transmission_fraction;
    }

    if (USE_RTR && !LAYERED_BRDF_FORCE_DIFFUSE_ONLY && push.debug_shading_mode != SHADING_MODE_RTX_OFF) {
        vec3 rtr_radiance;

        if (!RTR_RENDER_SCALED_BY_FG) {
            rtr_radiance = texelFetch(daxa_texture2D(rtr_tex), daxa_i32vec2(px), 0).xyz * brdf.energy_preservation.preintegrated_reflection;
        } else {
            rtr_radiance = texelFetch(daxa_texture2D(rtr_tex), daxa_i32vec2(px), 0).xyz;
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

    vec4 pt_cs = daxa_f32vec4(uv_to_cs(uv), depth, 1.0);
    vec4 pt_ws = deref(globals).player.cam.view_to_world * deref(globals).player.cam.sample_to_view * pt_cs;
    pt_ws /= pt_ws.w;
    pt_ws.xyz += deref(globals).player.player_unit_offset;
    uint rng = hash3(uvec3(px, deref(gpu_input).frame_index));
    if (FORCE_IRCACHE_DEBUG || push.debug_shading_mode == SHADING_MODE_IRCACHE) {
        IrcacheLookupParams ircache_params = IrcacheLookupParams_create(get_eye_position(globals), pt_ws.xyz, gbuffer.normal);
        output_ = lookup(ircache_params, rng);
    }

    output_ *= deref(gpu_input).pre_exposure;

    // output_ = gbuffer.albedo;
    imageStore(daxa_image2D(output_tex), daxa_i32vec2(px), daxa_f32vec4(output_, 1.0));
}

#endif

#if PostprocessingRasterShader

const mat3 SRGB_2_XYZ_MAT = mat3(
    0.4124564, 0.3575761, 0.1804375,
    0.2126729, 0.7151522, 0.0721750,
    0.0193339, 0.1191920, 0.9503041);
const float SRGB_ALPHA = 0.055;

vec3 srgb_encode(vec3 linear) {
    vec3 higher = (pow(abs(linear), vec3(0.41666666)) * (1.0 + SRGB_ALPHA)) - SRGB_ALPHA;
    vec3 lower = linear * 12.92;
    return mix(higher, lower, step(linear, vec3(0.0031308)));
}

daxa_f32 luminance(daxa_f32vec3 color) {
    daxa_f32vec3 luminanceCoefficients = SRGB_2_XYZ_MAT[1];
    return dot(color, luminanceCoefficients);
}

const daxa_f32mat3x3 agxTransform = daxa_f32mat3x3(
    0.842479062253094, 0.0423282422610123, 0.0423756549057051,
    0.0784335999999992, 0.878468636469772, 0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104);

const daxa_f32mat3x3 agxTransformInverse = daxa_f32mat3x3(
    1.19687900512017, -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433, 1.15107367264116);

daxa_f32vec3 agxDefaultContrastApproximation(daxa_f32vec3 x) {
    daxa_f32vec3 x2 = x * x;
    daxa_f32vec3 x4 = x2 * x2;

    return +15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

void agx(inout daxa_f32vec3 color) {
    const daxa_f32 minEv = -12.47393;
    const daxa_f32 maxEv = 4.026069;

    color = agxTransform * color;
    color = clamp(log2(color), minEv, maxEv);
    color = (color - minEv) / (maxEv - minEv);
    color = agxDefaultContrastApproximation(color);
}

void agxEotf(inout daxa_f32vec3 color) {
    color = agxTransformInverse * color;
}

void agxLook(inout daxa_f32vec3 color) {
    // Punchy
    const daxa_f32vec3 slope = daxa_f32vec3(1.1);
    const daxa_f32vec3 power = daxa_f32vec3(1.2);
    const daxa_f32 saturation = 1.3;

    daxa_f32 luma = luminance(color);

    color = pow(color * slope, power);
    color = max(luma + saturation * (color - luma), vec3(0.0));
}

const daxa_f32 exposureBias = 1.0;
const daxa_f32 calibration = 12.5;        // Light meter calibration
const daxa_f32 sensorSensitivity = 100.0; // Sensor sensitivity

daxa_f32 computeEV100fromLuminance(daxa_f32 luminance) {
    return log2(luminance * sensorSensitivity * exposureBias / calibration);
}

daxa_f32 computeExposureFromEV100(daxa_f32 ev100) {
    return 1.0 / (1.2 * exp2(ev100));
}

daxa_f32 computeExposure(daxa_f32 averageLuminance) {
    daxa_f32 ev100 = computeEV100fromLuminance(averageLuminance);
    daxa_f32 exposure = computeExposureFromEV100(ev100);

    return exposure;
}

daxa_f32vec3 color_correct(daxa_f32vec3 x) {
    agx(x);
    agxLook(x);
    agxEotf(x);
    // x = srgb_encode(x);
    return x;
}

layout(location = 0) out daxa_f32vec4 color;

void main() {
    daxa_f32vec2 g_buffer_scl = daxa_f32vec2(deref(gpu_input).render_res_scl) * daxa_f32vec2(deref(gpu_input).frame_dim) / daxa_f32vec2(deref(gpu_input).rounded_frame_dim);
    daxa_f32vec2 uv = daxa_f32vec2(gl_FragCoord.xy);
    daxa_f32vec3 final_color = texelFetch(daxa_texture2D(composited_image_id), daxa_i32vec2(uv), 0).rgb;

    if ((deref(gpu_input).flags & GAME_FLAG_BITS_PAUSED) == 0) {
        daxa_i32vec2 center_offset_uv = daxa_i32vec2(uv.xy) - daxa_i32vec2(deref(gpu_input).frame_dim.xy / deref(gpu_input).render_res_scl) / 2;
        if ((abs(center_offset_uv.x) <= 1 || abs(center_offset_uv.y) <= 1) && abs(center_offset_uv.x) + abs(center_offset_uv.y) < 6) {
            final_color *= daxa_f32vec3(0.1);
        }
        if ((abs(center_offset_uv.x) <= 0 || abs(center_offset_uv.y) <= 0) && abs(center_offset_uv.x) + abs(center_offset_uv.y) < 5) {
            final_color += daxa_f32vec3(2.0);
        }
    }

    color = daxa_f32vec4(color_correct(final_color), 1.0);
}

#endif

#if DebugImageRasterShader

layout(location = 0) out daxa_f32vec4 color;

void main() {
    daxa_f32vec2 uv = daxa_f32vec2(gl_FragCoord.xy) / daxa_f32vec2(push.output_tex_size.xy);
    daxa_f32vec3 tex_color;

    if (push.type == DEBUG_IMAGE_TYPE_GBUFFER) {
        daxa_i32vec2 in_pixel_i = daxa_i32vec2(uv * textureSize(daxa_utexture2D(image_id), 0).xy);
        daxa_u32vec4 g_buffer_value = texelFetch(daxa_utexture2D(image_id), in_pixel_i, 0);
        daxa_f32vec3 nrm = u16_to_nrm(g_buffer_value.y);
        daxa_f32 depth = uintBitsToFloat(g_buffer_value.z);
        tex_color = vec3(nrm);
        // tex_color = vec3(g_buffer_value.x * 0.00001, g_buffer_value.y * 0.0001, depth * 0.01);
    } else if (push.type == DEBUG_IMAGE_TYPE_SHADOW_BITMAP) {
        daxa_i32vec2 in_pixel_i = daxa_i32vec2(uv * textureSize(daxa_utexture2D(image_id), 0).xy);
        daxa_u32 shadow_value = texelFetch(daxa_utexture2D(image_id), in_pixel_i, 0).r;
        daxa_i32vec2 in_tile_i = daxa_i32vec2(uv * textureSize(daxa_utexture2D(image_id), 0).xy * daxa_f32vec2(8, 4)) & daxa_i32vec2(7, 3);
        daxa_u32 bit_index = in_tile_i.x + in_tile_i.y * 8;
        tex_color = vec3((shadow_value >> bit_index) & 1);
    } else if (push.type == DEBUG_IMAGE_TYPE_DEFAULT_UINT) {
        daxa_i32vec2 in_pixel_i = daxa_i32vec2(uv * textureSize(daxa_utexture2D(image_id), 0).xy);
        tex_color = texelFetch(daxa_utexture2D(image_id), in_pixel_i, 0).rgb;
    } else if (push.type == DEBUG_IMAGE_TYPE_DEFAULT) {
        daxa_i32vec2 in_pixel_i = daxa_i32vec2(uv * textureSize(daxa_texture2D(image_id), 0).xy);
        tex_color = texelFetch(daxa_texture2D(image_id), in_pixel_i, 0).rgb;
    } else if (push.type == DEBUG_IMAGE_TYPE_CUBEMAP) {
        uv = uv * vec2(3, 2);
        ivec2 uv_i = ivec2(floor(uv));
        uv = uv - uv_i;
        int face = uv_i.x + uv_i.y * 3;
        daxa_i32vec2 in_pixel_i = daxa_i32vec2(uv * push.cube_size);
        tex_color = texelFetch(daxa_texture2DArray(cube_image_id), ivec3(in_pixel_i, face), 0).rgb * 0.05;
    }

    color = daxa_f32vec4(tex_color, 1.0);
}

#endif

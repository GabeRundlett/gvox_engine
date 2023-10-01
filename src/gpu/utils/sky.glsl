#pragma once

#include <shared/core.inl>

#define SKY_INTENSITY 10
#define SUN_ANGULAR_DIAMETER 0.005
#define SUN_DIR deref(gpu_input).sky_settings.sun_direction
#define SUN_COL(sky_lut_tex) (get_far_sky_color(sky_lut_tex, SUN_DIR) * 5.0)

/* Return sqrt clamped to 0 */
f32 safe_sqrt(f32 x) { return sqrt(max(0, x)); }

f32 from_subuv_to_unit(f32 u, f32 resolution) {
    return (u - 0.5 / resolution) * (resolution / (resolution - 1.0));
}

f32 from_unit_to_subuv(f32 u, f32 resolution) {
    return (u + 0.5 / resolution) * (resolution / (resolution + 1.0));
}

/// Return distance of the first intersection between ray and sphere
/// @param r0 - ray origin
/// @param rd - normalized ray direction
/// @param s0 - sphere center
/// @param sR - sphere radius
/// @return return distance of intersection or -1.0 if there is no intersection
f32 ray_sphere_intersect_nearest(f32vec3 r0, f32vec3 rd, f32vec3 s0, f32 sR) {
    f32 a = dot(rd, rd);
    f32vec3 s0_r0 = r0 - s0;
    f32 b = 2.0 * dot(rd, s0_r0);
    f32 c = dot(s0_r0, s0_r0) - (sR * sR);
    f32 delta = b * b - 4.0 * a * c;
    if (delta < 0.0 || a == 0.0) {
        return -1.0;
    }
    f32 sol0 = (-b - safe_sqrt(delta)) / (2.0 * a);
    f32 sol1 = (-b + safe_sqrt(delta)) / (2.0 * a);
    if (sol0 < 0.0 && sol1 < 0.0) {
        return -1.0;
    }
    if (sol0 < 0.0) {
        return max(0.0, sol1);
    } else if (sol1 < 0.0) {
        return max(0.0, sol0);
    }
    return max(0.0, min(sol0, sol1));
}

struct SkyviewParams {
    f32 view_zenith_angle;
    f32 light_view_angle;
};

/// Get skyview LUT uv from skyview parameters
/// @param intersects_ground - true if ray intersects ground false otherwise
/// @param params - SkyviewParams structure
/// @param atmosphere_bottom - bottom of the atmosphere in km
/// @param atmosphere_top - top of the atmosphere in km
/// @param skyview_dimensions - skyViewLUT dimensions
/// @param view_height - view_height in world coordinates -> distance from planet center
/// @return - uv for the skyview LUT sampling
f32vec2 skyview_lut_params_to_uv(bool intersects_ground, SkyviewParams params,
                                 f32 atmosphere_bottom, f32 atmosphere_top, f32vec2 skyview_dimensions, f32 view_height) {
    f32vec2 uv;
    f32 beta = asin(atmosphere_bottom / view_height);
    f32 zenith_horizon_angle = PI - beta;

    if (!intersects_ground) {
        f32 coord = params.view_zenith_angle / zenith_horizon_angle;
        coord = (1.0 - safe_sqrt(1.0 - coord)) / 2.0;
        uv.y = coord;
    } else {
        f32 coord = (params.view_zenith_angle - zenith_horizon_angle) / beta;
        coord = (safe_sqrt(coord) + 1.0) / 2.0;
        uv.y = coord;
    }
    uv.x = safe_sqrt(params.light_view_angle / PI);
    uv = f32vec2(from_unit_to_subuv(uv.x, SKY_SKY_RES.x),
                 from_unit_to_subuv(uv.y, SKY_SKY_RES.y));
    return uv;
}

f32vec3 get_far_sky_color(daxa_ImageViewId a_sky_lut, f32vec3 world_direction) {
    // Because the atmosphere is using km as it's default units and we want one unit in world
    // space to be one meter we need to scale the position by a factor to get from meters -> kilometers
    const f32vec3 camera_position = f32vec3(0.0, 0.0, 0.1 + deref(gpu_input).sky_settings.atmosphere_bottom);

    const f32vec3 world_up = normalize(camera_position);

    const f32vec3 sun_direction = deref(gpu_input).sky_settings.sun_direction;
    const f32 view_zenith_angle = acos(dot(world_direction, world_up));
    const f32 light_view_angle = acos(dot(
        normalize(f32vec3(sun_direction.xy, 0.0)),
        normalize(f32vec3(world_direction.xy, 0.0))));

    const f32 atmosphere_intersection_distance = ray_sphere_intersect_nearest(
        camera_position,
        world_direction,
        f32vec3(0.0, 0.0, 0.0),
        deref(gpu_input).sky_settings.atmosphere_bottom);

    const bool intersects_ground = atmosphere_intersection_distance >= 0.0;
    const f32 camera_height = length(camera_position);

    f32vec2 skyview_uv = skyview_lut_params_to_uv(
        intersects_ground,
        SkyviewParams(view_zenith_angle, light_view_angle),
        deref(gpu_input).sky_settings.atmosphere_bottom,
        deref(gpu_input).sky_settings.atmosphere_top,
        f32vec2(SKY_SKY_RES),
        camera_height);

    f32vec3 sky_color = texture(daxa_sampler2D(a_sky_lut, deref(gpu_input).sampler_llc), skyview_uv).rgb;
    // if(!intersects_ground) { sky_color += add_sun_circle(world_direction, sun_direction); };

    return sky_color * SKY_INTENSITY;
}

f32vec3 get_far_sky_color_sun(daxa_ImageViewId a_sky_lut, f32vec3 nrm) {
    f32vec3 light = get_far_sky_color(a_sky_lut, nrm);
    f32 sun_val = dot(nrm, SUN_DIR) * 0.5 + 0.5;
    float x = cos(SUN_ANGULAR_DIAMETER);
    sun_val = (sun_val - x) / (1.0 - x);
    sun_val = clamp(sun_val * 15.0, 0, 15);
    light += sun_val * SUN_COL(a_sky_lut) * smoothstep(-0.0057, 0, dot(nrm, vec3(0, 0, 1)));
    return light;
}

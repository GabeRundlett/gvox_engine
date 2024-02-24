#pragma once

#include <application/input.inl>

#include "stars.glsl"
#include "sky_utils.glsl"

#include <g_samplers>

vec3 sun_radiance_impl(
    daxa_BufferPtr(GpuInput) gpu_input,
    daxa_ImageViewIndex _transmittance,
    vec3 view_direction,
    float height,
    float zenith_cos_angle,
    float sun_angular_radius_cos) {
    const vec3 sun_direction = deref(gpu_input).sky_settings.sun_direction;
    float cos_theta = dot(view_direction, sun_direction);

    if (cos_theta >= sun_angular_radius_cos) {
        TransmittanceParams transmittance_lut_params = TransmittanceParams(height, zenith_cos_angle);
        vec2 transmittance_texture_uv = transmittance_lut_to_uv(
            transmittance_lut_params,
            deref(gpu_input).sky_settings.atmosphere_bottom,
            deref(gpu_input).sky_settings.atmosphere_top);
        vec3 transmittance_to_sun = texture(
                                        daxa_sampler2D(_transmittance, g_sampler_llc),
                                        transmittance_texture_uv)
                                        .rgb;
        return transmittance_to_sun * sun_color.rgb * SUN_INTENSITY;
    } else {
        return vec3(0.0);
    }
}

vec3 get_atmosphere_radiance_along_ray(
    daxa_BufferPtr(GpuInput) gpu_input,
    daxa_ImageViewIndex _skyview,
    vec3 ray,
    float camera_height,
    vec3 sun_direction,
    out bool intersects_ground) {
    const vec3 world_camera_position = vec3(0, 0, camera_height);
    const vec3 world_up = vec3(0, 0, 1);

    const float view_zenith_angle = acos(dot(ray, world_up));
    // NOTE(grundlett): Minor imprecision in the dot-product can result in a value
    // just barely outside the valid range of acos (-1.0, 1.0). Sanity check and
    // clamp it.
    const float light_view_angle =
        acos(clamp(dot(normalize(vec3(sun_direction.xy, 0.0)),
                       normalize(vec3(ray.xy, 0.0))),
                   -1.0, 1.0));

    const float atmosphere_intersection_distance = ray_sphere_intersect_nearest(
        world_camera_position,
        ray,
        vec3(0.0, 0.0, 0.0),
        deref(gpu_input).sky_settings.atmosphere_bottom);

    intersects_ground = atmosphere_intersection_distance >= 0.0;
    bool inside_atmosphere = camera_height < deref(gpu_input).sky_settings.atmosphere_top;

    vec2 skyview_uv = skyview_lut_params_to_uv(
        inside_atmosphere,
        intersects_ground,
        SkyviewParams(view_zenith_angle, light_view_angle),
        deref(gpu_input).sky_settings.atmosphere_bottom,
        deref(gpu_input).sky_settings.atmosphere_top,
        vec2(SKY_SKY_RES.xy),
        camera_height);

    const vec3 unitless_atmosphere_illuminance = atmosphere_unpack(texture(daxa_sampler2D(_skyview, g_sampler_llc), skyview_uv));
    const vec3 sun_color_weighed_atmosphere_illuminance = sun_color.rgb * unitless_atmosphere_illuminance;
    const vec3 atmosphere_scattering_illuminance = sun_color_weighed_atmosphere_illuminance * SUN_INTENSITY;

    return atmosphere_scattering_illuminance;
}

// Sky represents everything, the atmosphere, sun, and stars.
vec3 sky_radiance_in_direction(daxa_BufferPtr(GpuInput) gpu_input, daxa_ImageViewIndex _skyview, daxa_ImageViewIndex _transmittance, vec3 view_direction) {
    vec3 world_camera_position = sky_space_camera_position(gpu_input);
    vec3 sun_direction = deref(gpu_input).sky_settings.sun_direction;
    float height = length(world_camera_position);

    const mat3 basis = build_orthonormal_basis(world_camera_position / height);
    world_camera_position = vec3(0, 0, height);
    view_direction = view_direction * basis;
    sun_direction = sun_direction * basis;

    bool normal_ray_intersects_ground;
    bool view_ray_intersects_ground;
    vec3 atmosphere_view_illuminance = get_atmosphere_radiance_along_ray(
        gpu_input,
        _skyview,
        view_direction,
        height,
        sun_direction,
        view_ray_intersects_ground);

    float zenith_cos_angle = dot(sun_direction, normalize(world_camera_position));
    float sun_angular_radius_cos = deref(gpu_input).sky_settings.sun_angular_radius_cos;

    const vec3 direct_sun_illuminance = view_ray_intersects_ground ? vec3(0.0) : sun_radiance_impl(gpu_input, _transmittance, view_direction, height, zenith_cos_angle, sun_angular_radius_cos);

    TransmittanceParams transmittance_lut_params = TransmittanceParams(height, dot(view_direction, normalize(world_camera_position)));
    vec2 transmittance_texture_uv = transmittance_lut_to_uv(
        transmittance_lut_params,
        deref(gpu_input).sky_settings.atmosphere_bottom,
        deref(gpu_input).sky_settings.atmosphere_top);
    vec3 atmosphere_transmittance = texture(daxa_sampler2D(_transmittance, g_sampler_llc), transmittance_texture_uv).rgb;

    const mat3 sun_basis = build_orthonormal_basis(normalize(SUN_DIRECTION));

    const float atmosphere_intersection_distance = ray_sphere_intersect_nearest(
        world_camera_position,
        view_direction,
        vec3(0.0, 0.0, 0.0),
        deref(gpu_input).sky_settings.atmosphere_top);

    bool intersects_sky = atmosphere_intersection_distance >= 0.0;
    if (!intersects_sky) {
        atmosphere_transmittance = vec3(1);
        atmosphere_view_illuminance = vec3(0);
    }

    return atmosphere_view_illuminance + direct_sun_illuminance + atmosphere_transmittance * get_star_radiance(gpu_input, (basis * view_direction) * sun_basis) * float(!view_ray_intersects_ground);
}

// Returns just the radiance from the sun in that direction
vec3 sun_radiance_in_direction(daxa_BufferPtr(GpuInput) gpu_input, daxa_ImageViewIndex transmittance_lut, vec3 nrm) {
    const vec3 world_camera_position = sky_space_camera_position(gpu_input);
    const vec3 sun_direction = deref(gpu_input).sky_settings.sun_direction;

    const float atmosphere_intersection_distance = ray_sphere_intersect_nearest(
        world_camera_position,
        nrm,
        vec3(0.0, 0.0, 0.0),
        deref(gpu_input).sky_settings.atmosphere_bottom);

    bool intersects_ground = atmosphere_intersection_distance >= 0.0;

    float height = length(world_camera_position);
    float zenith_cos_angle = dot(sun_direction, normalize(world_camera_position));
    float sun_angular_radius_cos = deref(gpu_input).sky_settings.sun_angular_radius_cos;

    const vec3 direct_sun_illuminance = intersects_ground ? vec3(0.0) : sun_radiance_impl(gpu_input, transmittance_lut, nrm, height, zenith_cos_angle, sun_angular_radius_cos);

    return direct_sun_illuminance;
}

#pragma once

#include <shared/core.inl>

#define SUN_DIRECTION deref(gpu_input).sky_settings.sun_direction
#define SUN_INTENSITY 1.0
const daxa_f32vec3 sun_color = daxa_f32vec3(255, 240, 233); // 5000 kelvin blackbody
#define SUN_COL(sky_lut_tex) (get_far_sky_color(sky_lut_tex, SUN_DIRECTION) * SUN_INTENSITY)
#define ATMOSPHERE_CAMERA_HEIGHT 3.0

daxa_f32vec3 get_sky_world_camera_position(daxa_BufferPtr(GpuInput) gpu_input) {
    // Because the atmosphere is using km as it's default units and we want one unit in world
    // space to be one meter we need to scale the position by a factor to get from meters -> kilometers
    const daxa_f32vec3 camera_position = daxa_f32vec3(0.0, 0.0, ATMOSPHERE_CAMERA_HEIGHT);
    daxa_f32vec3 world_camera_position = camera_position;
    world_camera_position.z += deref(gpu_input).sky_settings.atmosphere_bottom;
    return world_camera_position;
}

struct TransmittanceParams {
    daxa_f32 height;
    daxa_f32 zenith_cos_angle;
};

struct SkyviewParams {
    daxa_f32 view_zenith_angle;
    daxa_f32 light_view_angle;
};

/* Return sqrt clamped to 0 */
daxa_f32 safe_sqrt(daxa_f32 x) { return sqrt(max(0, x)); }

daxa_f32 from_subuv_to_unit(daxa_f32 u, daxa_f32 resolution) {
    return (u - 0.5 / resolution) * (resolution / (resolution - 1.0));
}

daxa_f32 from_unit_to_subuv(daxa_f32 u, daxa_f32 resolution) {
    return (u + 0.5 / resolution) * (resolution / (resolution + 1.0));
}

/// Return distance of the first intersection between ray and sphere
/// @param r0 - ray origin
/// @param rd - normalized ray direction
/// @param s0 - sphere center
/// @param sR - sphere radius
/// @return return distance of intersection or -1.0 if there is no intersection
daxa_f32 ray_sphere_intersect_nearest(daxa_f32vec3 r0, daxa_f32vec3 rd, daxa_f32vec3 s0, daxa_f32 sR) {
    daxa_f32 a = dot(rd, rd);
    daxa_f32vec3 s0_r0 = r0 - s0;
    daxa_f32 b = 2.0 * dot(rd, s0_r0);
    daxa_f32 c = dot(s0_r0, s0_r0) - (sR * sR);
    daxa_f32 delta = b * b - 4.0 * a * c;
    if (delta < 0.0 || a == 0.0) {
        return -1.0;
    }
    daxa_f32 sol0 = (-b - safe_sqrt(delta)) / (2.0 * a);
    daxa_f32 sol1 = (-b + safe_sqrt(delta)) / (2.0 * a);
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

const daxa_f32 PLANET_RADIUS_OFFSET = 0.01;

///	Transmittance LUT uses not uniform mapping -> transfer from mapping to texture uv
///	@param parameters
/// @param atmosphere_bottom - bottom radius of the atmosphere in km
/// @param atmosphere_top - top radius of the atmosphere in km
///	@return - uv of the corresponding texel
daxa_f32vec2 transmittance_lut_to_uv(TransmittanceParams parameters, daxa_f32 atmosphere_bottom, daxa_f32 atmosphere_top) {
    daxa_f32 H = safe_sqrt(atmosphere_top * atmosphere_top - atmosphere_bottom * atmosphere_bottom);
    daxa_f32 rho = safe_sqrt(parameters.height * parameters.height - atmosphere_bottom * atmosphere_bottom);

    daxa_f32 discriminant = parameters.height * parameters.height *
                                (parameters.zenith_cos_angle * parameters.zenith_cos_angle - 1.0) +
                            atmosphere_top * atmosphere_top;
    /* Distance to top atmosphere boundary */
    daxa_f32 d = max(0.0, (-parameters.height * parameters.zenith_cos_angle + safe_sqrt(discriminant)));

    daxa_f32 d_min = atmosphere_top - parameters.height;
    daxa_f32 d_max = rho + H;
    daxa_f32 mu = (d - d_min) / (d_max - d_min);
    daxa_f32 r = rho / H;

    return daxa_f32vec2(mu, r);
}

/// Transmittance LUT uses not uniform mapping -> transfer from uv to this mapping
/// @param uv - uv in the range [0,1]
/// @param atmosphere_bottom - bottom radius of the atmosphere in km
/// @param atmosphere_top - top radius of the atmosphere in km
/// @return - TransmittanceParams structure
TransmittanceParams uv_to_transmittance_lut_params(daxa_f32vec2 uv, daxa_f32 atmosphere_bottom, daxa_f32 atmosphere_top) {
    TransmittanceParams params;
    daxa_f32 H = safe_sqrt(atmosphere_top * atmosphere_top - atmosphere_bottom * atmosphere_bottom.x);

    daxa_f32 rho = H * uv.y;
    params.height = safe_sqrt(rho * rho + atmosphere_bottom * atmosphere_bottom);

    daxa_f32 d_min = atmosphere_top - params.height;
    daxa_f32 d_max = rho + H;
    daxa_f32 d = d_min + uv.x * (d_max - d_min);

    params.zenith_cos_angle = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * params.height * d);
    params.zenith_cos_angle = clamp(params.zenith_cos_angle, -1.0, 1.0);

    return params;
}

/// Get parameters used for skyview LUT computation from uv coords
/// @param uv - texel uv in the range [0,1]
/// @param atmosphere_bottom - bottom of the atmosphere in km
/// @param atmosphere_top - top of the atmosphere in km
/// @param skyview dimensions
/// @param view_height - view_height in world coordinates -> distance from planet center
/// @return - SkyviewParams structure
SkyviewParams uv_to_skyview_lut_params(daxa_f32vec2 uv, daxa_f32 atmosphere_bottom,
                                       daxa_f32 atmosphere_top, daxa_f32vec2 skyview_dimensions, daxa_f32 view_height) {
    /* Constrain uvs to valid sub texel range
    (avoid zenith derivative issue making LUT usage visible) */
    uv = daxa_f32vec2(from_subuv_to_unit(uv.x, skyview_dimensions.x),
                      from_subuv_to_unit(uv.y, skyview_dimensions.y));

    daxa_f32 beta = asin(atmosphere_bottom / view_height);
    daxa_f32 zenith_horizon_angle = M_PI - beta;

    daxa_f32 view_zenith_angle;
    daxa_f32 light_view_angle;
    /* Nonuniform mapping near the horizon to avoid artefacts */
    if (uv.y < 0.5) {
        daxa_f32 coord = 1.0 - (1.0 - 2.0 * uv.y) * (1.0 - 2.0 * uv.y);
        view_zenith_angle = zenith_horizon_angle * coord;
    } else {
        daxa_f32 coord = (uv.y * 2.0 - 1.0) * (uv.y * 2.0 - 1.0);
        view_zenith_angle = zenith_horizon_angle + beta * coord;
    }
    light_view_angle = (uv.x * uv.x) * M_PI;
    return SkyviewParams(view_zenith_angle, light_view_angle);
}

/// Moves to the nearest intersection with top of the atmosphere in the direction specified in
/// world_direction
/// @param world_position - current world position -> will be changed to new pos at the top of
/// 		the atmosphere if there exists such intersection
/// @param world_direction - the direction in which the shift will be done
/// @param atmosphere_bottom - bottom of the atmosphere in km
/// @param atmosphere_top - top of the atmosphere in km
daxa_b32 move_to_top_atmosphere(inout daxa_f32vec3 world_position, daxa_f32vec3 world_direction,
                                daxa_f32 atmosphere_bottom, daxa_f32 atmosphere_top) {
    daxa_f32vec3 planet_origin = daxa_f32vec3(0.0, 0.0, 0.0);
    /* Check if the world_position is outside of the atmosphere */
    if (length(world_position) > atmosphere_top) {
        daxa_f32 dist_to_top_atmo_intersection = ray_sphere_intersect_nearest(
            world_position, world_direction, planet_origin, atmosphere_top);

        /* No intersection with the atmosphere */
        if (dist_to_top_atmo_intersection == -1.0) {
            return false;
        } else {
            daxa_f32vec3 up_offset = normalize(world_position) * -PLANET_RADIUS_OFFSET;
            world_position += world_direction * dist_to_top_atmo_intersection + up_offset;
        }
    }
    /* Position is in or at the top of the atmosphere */
    return true;
}

/// @param params - buffer reference to the atmosphere parameters buffer
/// @param position - position in the world where the sample is to be taken
/// @return atmosphere extinction at the desired point
daxa_f32vec3 sample_medium_extinction(daxa_BufferPtr(GpuInput) gpu_input, daxa_f32vec3 position) {
    const daxa_f32 height = length(position) - deref(gpu_input).sky_settings.atmosphere_bottom;

    const daxa_f32 density_mie = exp(deref(gpu_input).sky_settings.mie_density[1].exp_scale * height);
    const daxa_f32 density_ray = exp(deref(gpu_input).sky_settings.rayleigh_density[1].exp_scale * height);
    // const daxa_f32 density_ozo = clamp(height < deref(gpu_input).sky_settings.absorption_density[0].layer_width ?
    //     deref(gpu_input).sky_settings.absorption_density[0].lin_term * height + deref(gpu_input).sky_settings.absorption_density[0].const_term :
    //     deref(gpu_input).sky_settings.absorption_density[1].lin_term * height + deref(gpu_input).sky_settings.absorption_density[1].const_term,
    //     0.0, 1.0);
    const daxa_f32 density_ozo = exp(-max(0.0, 35.0 - height) * (1.0 / 5.0)) * exp(-max(0.0, height - 35.0) * (1.0 / 15.0)) * 2;
    daxa_f32vec3 mie_extinction = deref(gpu_input).sky_settings.mie_extinction * density_mie;
    daxa_f32vec3 ray_extinction = deref(gpu_input).sky_settings.rayleigh_scattering * density_ray;
    daxa_f32vec3 ozo_extinction = deref(gpu_input).sky_settings.absorption_extinction * density_ozo;

    return mie_extinction + ray_extinction + ozo_extinction;
}

/// @param params - buffer reference to the atmosphere parameters buffer
/// @param position - position in the world where the sample is to be taken
/// @return atmosphere scattering at the desired point
daxa_f32vec3 sample_medium_scattering(daxa_BufferPtr(GpuInput) gpu_input, daxa_f32vec3 position) {
    const daxa_f32 height = length(position) - deref(gpu_input).sky_settings.atmosphere_bottom;

    const daxa_f32 density_mie = exp(deref(gpu_input).sky_settings.mie_density[1].exp_scale * height);
    const daxa_f32 density_ray = exp(deref(gpu_input).sky_settings.rayleigh_density[1].exp_scale * height);

    daxa_f32vec3 mie_scattering = deref(gpu_input).sky_settings.mie_scattering * density_mie;
    daxa_f32vec3 ray_scattering = deref(gpu_input).sky_settings.rayleigh_scattering * density_ray;
    /* Not considering ozon scattering in current version of this model */
    daxa_f32vec3 ozo_scattering = daxa_f32vec3(0.0, 0.0, 0.0);

    return mie_scattering + ray_scattering + ozo_scattering;
}

struct ScatteringSample {
    daxa_f32vec3 mie;
    daxa_f32vec3 ray;
};
/// @param params - buffer reference to the atmosphere parameters buffer
/// @param position - position in the world where the sample is to be taken
/// @return Scattering sample struct
// TODO(msakmary) Fix this!!
ScatteringSample sample_medium_scattering_detailed(daxa_BufferPtr(GpuInput) gpu_input, daxa_f32vec3 position) {
    const daxa_f32 height = length(position) - deref(gpu_input).sky_settings.atmosphere_bottom;

    const daxa_f32 density_mie = exp(deref(gpu_input).sky_settings.mie_density[1].exp_scale * height);
    const daxa_f32 density_ray = exp(deref(gpu_input).sky_settings.rayleigh_density[1].exp_scale * height);
    const daxa_f32 density_ozo = clamp(height < deref(gpu_input).sky_settings.absorption_density[0].layer_width ? deref(gpu_input).sky_settings.absorption_density[0].lin_term * height + deref(gpu_input).sky_settings.absorption_density[0].const_term : deref(gpu_input).sky_settings.absorption_density[1].lin_term * height + deref(gpu_input).sky_settings.absorption_density[1].const_term,
                                       0.0, 1.0);

    daxa_f32vec3 mie_scattering = deref(gpu_input).sky_settings.mie_scattering * density_mie;
    daxa_f32vec3 ray_scattering = deref(gpu_input).sky_settings.rayleigh_scattering * density_ray;
    /* Not considering ozon scattering in current version of this model */
    daxa_f32vec3 ozo_scattering = daxa_f32vec3(0.0, 0.0, 0.0);

    return ScatteringSample(mie_scattering, ray_scattering);
}

/// Get skyview LUT uv from skyview parameters
/// @param intersects_ground - true if ray intersects ground false otherwise
/// @param params - SkyviewParams structure
/// @param atmosphere_bottom - bottom of the atmosphere in km
/// @param atmosphere_top - top of the atmosphere in km
/// @param skyview_dimensions - skyViewLUT dimensions
/// @param view_height - view_height in world coordinates -> distance from planet center
/// @return - uv for the skyview LUT sampling
daxa_f32vec2 skyview_lut_params_to_uv(bool intersects_ground, SkyviewParams params,
                                      daxa_f32 atmosphere_bottom, daxa_f32 atmosphere_top, daxa_f32vec2 skyview_dimensions, daxa_f32 view_height) {
    daxa_f32vec2 uv;
    daxa_f32 beta = asin(atmosphere_bottom / view_height);
    daxa_f32 zenith_horizon_angle = M_PI - beta;

    if (!intersects_ground) {
        daxa_f32 coord = params.view_zenith_angle / zenith_horizon_angle;
        coord = (1.0 - safe_sqrt(1.0 - coord)) / 2.0;
        uv.y = coord;
    } else {
        daxa_f32 coord = (params.view_zenith_angle - zenith_horizon_angle) / beta;
        coord = (safe_sqrt(coord) + 1.0) / 2.0;
        uv.y = coord;
    }
    uv.x = safe_sqrt(params.light_view_angle / M_PI);
    uv = daxa_f32vec2(from_unit_to_subuv(uv.x, SKY_SKY_RES.x),
                      from_unit_to_subuv(uv.y, SKY_SKY_RES.y));
    return uv;
}

daxa_f32vec3 get_sun_illuminance(
    daxa_BufferPtr(GpuInput) gpu_input,
    daxa_ImageViewIndex _transmittance,
    daxa_f32vec3 view_direction,
    daxa_f32 height,
    daxa_f32 zenith_cos_angle,
    daxa_f32 sun_angular_radius_cos) {
    const daxa_f32vec3 sun_direction = deref(gpu_input).sky_settings.sun_direction;
    daxa_f32 cos_theta = dot(view_direction, sun_direction);

    if (cos_theta >= sun_angular_radius_cos) {
        TransmittanceParams transmittance_lut_params = TransmittanceParams(height, zenith_cos_angle);
        daxa_f32vec2 transmittance_texture_uv = transmittance_lut_to_uv(
            transmittance_lut_params,
            deref(gpu_input).sky_settings.atmosphere_bottom,
            deref(gpu_input).sky_settings.atmosphere_top);
        daxa_f32vec3 transmittance_to_sun = texture(
                                                daxa_sampler2D(_transmittance, deref(gpu_input).sampler_llc),
                                                transmittance_texture_uv)
                                                .rgb;
        return transmittance_to_sun * sun_color.rgb * SUN_INTENSITY;
    } else {
        return daxa_f32vec3(0.0);
    }
}

daxa_f32vec3 get_atmosphere_illuminance_along_ray(
    daxa_BufferPtr(GpuInput) gpu_input,
    daxa_ImageViewIndex _skyview,
    daxa_f32vec3 ray,
    daxa_f32vec3 world_camera_position,
    daxa_f32vec3 sun_direction,
    out bool intersects_ground) {
    const daxa_f32vec3 world_up = normalize(world_camera_position);

    const daxa_f32 view_zenith_angle = acos(dot(ray, world_up));
    const daxa_f32 light_view_angle = acos(dot(
        normalize(daxa_f32vec3(sun_direction.xy, 0.0)),
        normalize(daxa_f32vec3(ray.xy, 0.0))));

    const daxa_f32 atmosphere_intersection_distance = ray_sphere_intersect_nearest(
        world_camera_position,
        ray,
        daxa_f32vec3(0.0, 0.0, 0.0),
        deref(gpu_input).sky_settings.atmosphere_bottom);

    intersects_ground = atmosphere_intersection_distance >= 0.0;
    const daxa_f32 camera_height = length(world_camera_position);

    daxa_f32vec2 skyview_uv = skyview_lut_params_to_uv(
        intersects_ground,
        SkyviewParams(view_zenith_angle, light_view_angle),
        deref(gpu_input).sky_settings.atmosphere_bottom,
        deref(gpu_input).sky_settings.atmosphere_top,
        daxa_f32vec2(SKY_SKY_RES.xy),
        camera_height);

    const daxa_f32vec3 unitless_atmosphere_illuminance = texture(daxa_sampler2D(_skyview, deref(gpu_input).sampler_llc), skyview_uv).rgb;
    const daxa_f32vec3 sun_color_weighed_atmosphere_illuminance = sun_color.rgb * unitless_atmosphere_illuminance;
    const daxa_f32vec3 atmosphere_scattering_illuminance = sun_color_weighed_atmosphere_illuminance * SUN_INTENSITY;

    return atmosphere_scattering_illuminance;
}

struct AtmosphereLightingInfo {
    // illuminance from atmosphere along normal vector
    daxa_f32vec3 atmosphere_normal_illuminance;
    // illuminance from atmosphere along view vector
    daxa_f32vec3 atmosphere_direct_illuminance;
    // direct sun illuminance
    daxa_f32vec3 sun_direct_illuminance;
};

AtmosphereLightingInfo get_atmosphere_lighting(daxa_BufferPtr(GpuInput) gpu_input, daxa_ImageViewIndex _skyview, daxa_ImageViewIndex _transmittance, daxa_f32vec3 view_direction, daxa_f32vec3 normal) {
    const daxa_f32vec3 world_camera_position = get_sky_world_camera_position(gpu_input);
    const daxa_f32vec3 sun_direction = deref(gpu_input).sky_settings.sun_direction;

    bool normal_ray_intersects_ground;
    bool view_ray_intersects_ground;
    const daxa_f32vec3 atmosphere_normal_illuminance = get_atmosphere_illuminance_along_ray(
        gpu_input,
        _skyview,
        normal,
        world_camera_position,
        sun_direction,
        normal_ray_intersects_ground);
    const daxa_f32vec3 atmosphere_view_illuminance = get_atmosphere_illuminance_along_ray(
        gpu_input,
        _skyview,
        view_direction,
        world_camera_position,
        sun_direction,
        view_ray_intersects_ground);

    const daxa_f32vec3 direct_sun_illuminance = view_ray_intersects_ground ? daxa_f32vec3(0.0) : get_sun_illuminance(gpu_input, _transmittance, view_direction, length(world_camera_position), dot(sun_direction, normalize(world_camera_position)), deref(gpu_input).sky_settings.sun_angular_radius_cos);

    return AtmosphereLightingInfo(
        atmosphere_normal_illuminance,
        atmosphere_view_illuminance,
        direct_sun_illuminance);
}

vec3 sample_sun_direction(
    daxa_BufferPtr(GpuInput) gpu_input,
    vec2 urand, bool soft) {
    if (soft && PER_VOXEL_NORMALS == 0) {
        float sun_angular_radius_cos = deref(gpu_input).sky_settings.sun_angular_radius_cos;
        if (sun_angular_radius_cos < 1.0) {
            const mat3 basis = build_orthonormal_basis(normalize(SUN_DIRECTION));
            return basis * uniform_sample_cone(urand, sun_angular_radius_cos);
        }
    }
    return SUN_DIRECTION;
}

vec3 sun_color_in_direction(
    daxa_BufferPtr(GpuInput) gpu_input,
    daxa_ImageViewIndex transmittance_lut, vec3 nrm) {
    const daxa_f32vec3 world_camera_position = get_sky_world_camera_position(gpu_input);
    const daxa_f32vec3 sun_direction = deref(gpu_input).sky_settings.sun_direction;

    const daxa_f32 atmosphere_intersection_distance = ray_sphere_intersect_nearest(
        world_camera_position,
        nrm,
        daxa_f32vec3(0.0, 0.0, 0.0),
        deref(gpu_input).sky_settings.atmosphere_bottom);

    bool intersects_ground = atmosphere_intersection_distance >= 0.0;
    const daxa_f32vec3 direct_sun_illuminance = intersects_ground ? daxa_f32vec3(0.0) : get_sun_illuminance(gpu_input, transmittance_lut, nrm, length(world_camera_position), dot(sun_direction, normalize(world_camera_position)), deref(gpu_input).sky_settings.sun_angular_radius_cos);

    return direct_sun_illuminance;
}

#pragma once

#include <application/input.inl>

/// @param params - buffer reference to the atmosphere parameters buffer
/// @param position - position in the world where the sample is to be taken
/// @return atmosphere extinction at the desired point
vec3 sample_medium_extinction(daxa_BufferPtr(GpuInput) gpu_input, vec3 position) {
    const float height = length(position) - deref(gpu_input).sky_settings.atmosphere_bottom;

    const float density_mie = exp(deref(gpu_input).sky_settings.mie_density[1].exp_scale * height);
    const float density_ray = exp(deref(gpu_input).sky_settings.rayleigh_density[1].exp_scale * height);
    // const float density_ozo = clamp(height < deref(gpu_input).sky_settings.absorption_density[0].layer_width ?
    //     deref(gpu_input).sky_settings.absorption_density[0].lin_term * height + deref(gpu_input).sky_settings.absorption_density[0].const_term :
    //     deref(gpu_input).sky_settings.absorption_density[1].lin_term * height + deref(gpu_input).sky_settings.absorption_density[1].const_term,
    //     0.0, 1.0);
    const float density_ozo = exp(-max(0.0, 35.0 - height) * (1.0 / 5.0)) * exp(-max(0.0, height - 35.0) * (1.0 / 15.0)) * 2;
    vec3 mie_extinction = deref(gpu_input).sky_settings.mie_extinction * max(density_mie, 0.0);
    vec3 ray_extinction = deref(gpu_input).sky_settings.rayleigh_scattering * max(density_ray, 0.0);
    vec3 ozo_extinction = deref(gpu_input).sky_settings.absorption_extinction * max(density_ozo, 0.0);

    return mie_extinction + ray_extinction + ozo_extinction;
}

/// @param params - buffer reference to the atmosphere parameters buffer
/// @param position - position in the world where the sample is to be taken
/// @return atmosphere scattering at the desired point
vec3 sample_medium_scattering(daxa_BufferPtr(GpuInput) gpu_input, vec3 position) {
    const float height = length(position) - deref(gpu_input).sky_settings.atmosphere_bottom;

    const float density_mie = exp(deref(gpu_input).sky_settings.mie_density[1].exp_scale * height);
    const float density_ray = exp(deref(gpu_input).sky_settings.rayleigh_density[1].exp_scale * height);

    vec3 mie_scattering = deref(gpu_input).sky_settings.mie_scattering * max(density_mie, 0.0);
    vec3 ray_scattering = deref(gpu_input).sky_settings.rayleigh_scattering * max(density_ray, 0.0);
    /* Not considering ozon scattering in current version of this model */
    vec3 ozo_scattering = vec3(0.0, 0.0, 0.0);

    return mie_scattering + ray_scattering + ozo_scattering;
}

struct ScatteringSample {
    vec3 mie;
    vec3 ray;
};
/// @param params - buffer reference to the atmosphere parameters buffer
/// @param position - position in the world where the sample is to be taken
/// @return Scattering sample struct
// TODO(msakmary) Fix this!!
ScatteringSample sample_medium_scattering_detailed(daxa_BufferPtr(GpuInput) gpu_input, vec3 position) {
    const float height = length(position) - deref(gpu_input).sky_settings.atmosphere_bottom;

    const float density_mie = exp(deref(gpu_input).sky_settings.mie_density[1].exp_scale * height);
    const float density_ray = exp(deref(gpu_input).sky_settings.rayleigh_density[1].exp_scale * height);

    vec3 mie_scattering = deref(gpu_input).sky_settings.mie_scattering * max(density_mie, 0.0);
    vec3 ray_scattering = deref(gpu_input).sky_settings.rayleigh_scattering * max(density_ray, 0.0);
    /* Not considering ozon scattering in current version of this model */
    vec3 ozo_scattering = vec3(0.0, 0.0, 0.0);

    return ScatteringSample(mie_scattering, ray_scattering);
}

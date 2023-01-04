#pragma once

#define DISABLE_SKY 0
#define USE_OLD_SKY 0
#define USE_BAKED_OPTICAL_DEPTH 1

#define SKY_COL (f32vec3(0.02, 0.05, 0.90))
#define SKY_COL_B (f32vec3(0.08, 0.10, 0.54))

#define SUN_TIME INPUT.settings.daylight_cycle_time
#define SUN_DIR normalize(f32vec3(0.8 * sin(SUN_TIME), 2.3 * cos(SUN_TIME), sin(SUN_TIME)))
#if DISABLE_SKY
#define SUN_COL f32vec3(1, 1, 1)
#define INTERNAL_SUN_FACTOR 1
#define SUN_FACTOR 1
#else
#define SUN_COL (f32vec3(1, 0.85, 0.5) * 2)
#define INTERNAL_SUN_FACTOR (1 - pow(1 - max(sin(SUN_TIME), 0.0005), 2))
#define SUN_FACTOR (1 - pow(1 - clamp(sin(SUN_TIME) + 0.1, 0.0001, 1.0), 50))
#endif

#define PLANET_RADIUS 0.95
#define ATMOSPHERE_RADIUS 1.0
#define ATMOSPHERE_DENSITY_FALLOFF 4.0
#define WAVELENGTH_X 700.0
#define WAVELENGTH_Y 530.0
#define WAVELENGTH_Z 440.0
#define SCATTERING_STRENGTH 30.0
#define SCATTERING_COEFF (f32vec3(pow(400.0 / WAVELENGTH_X, 4), pow(400.0 / WAVELENGTH_Y, 4), pow(400.0 / WAVELENGTH_Z, 4)) * SCATTERING_STRENGTH)

#define IN_SCATTER_N 4
#define OPTICAL_DEPTH_N 100

f32 calc_atmosphere_depth(Ray ray) {
    Sphere atmo_sphere;
    atmo_sphere.o = f32vec3(0, 0, 0);
    atmo_sphere.r = ATMOSPHERE_RADIUS;
    IntersectionRecord atmo_sphere_intersection = intersect(ray, atmo_sphere);
    return atmo_sphere_intersection.dist;
}
f32 calc_atmosphere_density(f32vec3 p) {
    f32 height_above_surface = length(p) - PLANET_RADIUS;
    f32 height_normalized = height_above_surface / (ATMOSPHERE_RADIUS - PLANET_RADIUS);
    f32 local_density = exp(-height_normalized * ATMOSPHERE_DENSITY_FALLOFF) * (1.0 - height_normalized);
    return local_density;
}
f32 optical_depth(Ray ray, f32 ray_length) {
    f32vec3 density_sample_p = ray.o;
    f32 step_size = ray_length / f32(OPTICAL_DEPTH_N - 1);
    f32 result = 0;
    for (u32 i = 0; i < OPTICAL_DEPTH_N; ++i) {
        f32 local_density = calc_atmosphere_density(density_sample_p);
        result += local_density * step_size;
        density_sample_p += ray.nrm * step_size;
    }
    return result;
}

#if !defined(SKY_ONLY_OPTICAL_DEPTH)

f32 optical_depth_baked(Ray ray) {
    f32 height_above_surface = length(ray.o) - PLANET_RADIUS;
    f32 height_normalized = clamp(height_above_surface / (ATMOSPHERE_RADIUS - PLANET_RADIUS), 0, 1);
    f32 angle = 1.0 - (dot(normalize(ray.o), ray.nrm) * 0.5 + 0.5);
    // clang-format off
    return texture(sampler2D(
        get_texture(texture2D, daxa_push_constant.optical_depth_image_id),
        get_sampler(daxa_push_constant.optical_depth_sampler_id)),
        f32vec2(angle, height_normalized)).r;
    // clang-format on
}

f32 optical_depth_baked2(Ray ray, f32 ray_length) {
    f32vec3 end_p = ray.o + ray.nrm * ray_length;
    f32 d = dot(ray.nrm, normalize(ray.o));
    f32 optical_depth = 0;
    Ray ray00 = Ray(ray.o, +ray.nrm, +ray.inv_nrm);
    Ray ray01 = Ray(ray.o, -ray.nrm, -ray.inv_nrm);
    Ray ray10 = Ray(end_p, +ray.nrm, +ray.inv_nrm);
    Ray ray11 = Ray(end_p, -ray.nrm, -ray.inv_nrm);
    f32 w = clamp(d * 1.5 + 0.5, 0, 1);
    f32 d0 = optical_depth_baked(ray00) - optical_depth_baked(ray10);
    f32 d1 = optical_depth_baked(ray11) - optical_depth_baked(ray01);
    return mix(d1, d0, w);
}

f32vec3 calc_light(Ray ray, f32 ray_length) {
    f32vec3 in_scatter_p = ray.o;
    f32 step_size = ray_length / f32(IN_SCATTER_N - 1);
    f32vec3 result = f32vec3(0, 0, 0);
    for (u32 i = 0; i < IN_SCATTER_N; ++i) {
        Ray sun_ray = Ray(in_scatter_p, SUN_DIR, 1.0 / SUN_DIR);
        f32 sun_ray_length = calc_atmosphere_depth(sun_ray);

#if USE_BAKED_OPTICAL_DEPTH
        f32 sun_ray_optical_depth = optical_depth_baked(sun_ray);
#else
        f32 sun_ray_optical_depth = optical_depth(sun_ray, sun_ray_length);
#endif

        Ray ray_to_view;
        ray_to_view.o = sun_ray.o;
        ray_to_view.nrm = -ray.nrm;
        ray_to_view.inv_nrm = 1.0 / ray_to_view.nrm;
#if USE_BAKED_OPTICAL_DEPTH
        f32 view_ray_optical_depth = optical_depth_baked2(ray_to_view, step_size * i);
#else
        f32 view_ray_optical_depth = optical_depth(ray_to_view, step_size * i);
#endif
        f32vec3 transmittance = exp(-(sun_ray_optical_depth + view_ray_optical_depth) * SCATTERING_COEFF);
        f32 local_density = calc_atmosphere_density(sun_ray.o);
        result += local_density * transmittance * step_size;
        in_scatter_p += ray.nrm * step_size;
    }

    f32 fac = dot(f32vec3(0, 0, 1), ray.nrm) * 0.4 + 0.4;
    fac = (fac - 0.25) * 2;
    fac = clamp(fac, 0, 1);
    result = mix(f32vec3(0.2, 0.2, 0.2) * INTERNAL_SUN_FACTOR, result, fac);

    return result + SKY_COL * INTERNAL_SUN_FACTOR;
}

f32vec3 sample_sky_ambient(f32vec3 nrm) {
#if DISABLE_SKY
    return f32vec3(0.5);
#elif USE_OLD_SKY
    f32 sun_val = dot(nrm, SUN_DIR) * 0.25 + 0.5;
    sun_val *= f32(dot(nrm, f32vec3(0, 0, 1)) > 0.0);
    sun_val = pow(sun_val, 2) * 0.2;
    f32 sky_val = clamp(dot(nrm, f32vec3(0, 0, -1)) * 0.5 + 0.5, 0, 1);
    return mix(SKY_COL + sun_val * SUN_COL, SKY_COL_B, pow(sky_val, 2)) * INTERNAL_SUN_FACTOR;
#else
    Ray ray;
    ray.o = f32vec3(0, 0, PLANET_RADIUS);
    ray.nrm = nrm;
    ray.inv_nrm = 1.0 / ray.nrm;
    const f32 epsilon = 0.001;
    f32 atmosphere_depth = calc_atmosphere_depth(ray) + epsilon;
    f32vec3 light = calc_light(ray, atmosphere_depth - epsilon * 2);
    return light;
#endif
}

f32vec3 sample_sky(f32vec3 nrm) {
    f32vec3 light = sample_sky_ambient(nrm);
#if !DISABLE_SKY
    f32 sun_val = dot(nrm, SUN_DIR) * 0.5 + 0.5;
    sun_val *= f32(dot(nrm, f32vec3(0, 0, 1)) > 0.0);
    sun_val = sun_val * 1000 - 999;
    sun_val = pow(clamp(sun_val * 1.5, 0, 1), 20);
    light += sun_val * SUN_COL * 1.0;
#endif
    return light;
}

#endif

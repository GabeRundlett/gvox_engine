#include <renderer/sky.inl>

#include <utils/math.glsl>
#include <utils/sky.glsl>

#if SkyTransmittanceComputeShader

DAXA_DECL_PUSH_CONSTANT(SkyTransmittanceComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewIndex transmittance_lut = push.uses.transmittance_lut;

layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;
daxa_f32vec3 integrate_transmittance(daxa_f32vec3 world_position, daxa_f32vec3 world_direction, daxa_u32 sample_count) {
    /* The length of ray between position and nearest atmosphere top boundary */
    daxa_f32 integration_length = ray_sphere_intersect_nearest(
        world_position,
        world_direction,
        daxa_f32vec3(0.0, 0.0, 0.0),
        deref(gpu_input).sky_settings.atmosphere_top);

    daxa_f32 integration_step = integration_length / daxa_f32(sample_count);

    /* Result of the integration */
    daxa_f32vec3 optical_depth = daxa_f32vec3(0.0, 0.0, 0.0);

    for (daxa_i32 i = 0; i < sample_count; i++) {
        /* Move along the world direction ray to new position */
        daxa_f32vec3 new_pos = world_position + i * integration_step * world_direction;
        daxa_f32vec3 atmosphere_extinction = sample_medium_extinction(gpu_input, new_pos);
        optical_depth += atmosphere_extinction * integration_step;
    }
    return optical_depth;
}

void main() {
    if (any(greaterThan(gl_GlobalInvocationID.xy, SKY_TRANSMITTANCE_RES))) {
        return;
    }

    daxa_f32vec2 uv = daxa_f32vec2(gl_GlobalInvocationID.xy) / daxa_f32vec2(SKY_TRANSMITTANCE_RES);

    TransmittanceParams mapping = uv_to_transmittance_lut_params(
        uv,
        deref(gpu_input).sky_settings.atmosphere_bottom,
        deref(gpu_input).sky_settings.atmosphere_top);

    daxa_f32vec3 world_position = daxa_f32vec3(0.0, 0.0, mapping.height);
    daxa_f32vec3 world_direction = daxa_f32vec3(
        safe_sqrt(1.0 - mapping.zenith_cos_angle * mapping.zenith_cos_angle),
        0.0,
        mapping.zenith_cos_angle);

    daxa_f32vec3 transmittance = exp(-integrate_transmittance(world_position, world_direction, 400));

    imageStore(daxa_image2D(transmittance_lut), daxa_i32vec2(gl_GlobalInvocationID.xy), daxa_f32vec4(transmittance, 1.0));
}

#endif

#if SkyMultiscatteringComputeShader

DAXA_DECL_PUSH_CONSTANT(SkyMultiscatteringComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewIndex transmittance_lut = push.uses.transmittance_lut;
daxa_ImageViewIndex multiscattering_lut = push.uses.multiscattering_lut;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 64) in;
/* This number should match the number of local threads -> z dimension */
const daxa_f32 SPHERE_SAMPLES = 64.0;
const daxa_f32 GOLDEN_RATIO = 1.6180339;
const daxa_f32 uniformPhase = 1.0 / (4.0 * M_PI);

shared daxa_f32vec3 multiscatt_shared[64];
shared daxa_f32vec3 luminance_shared[64];

struct RaymarchResult {
    daxa_f32vec3 luminance;
    daxa_f32vec3 multiscattering;
};

RaymarchResult integrate_scattered_luminance(daxa_f32vec3 world_position, daxa_f32vec3 world_direction, daxa_f32vec3 sun_direction, daxa_f32 sample_count) {
    RaymarchResult result = RaymarchResult(daxa_f32vec3(0.0, 0.0, 0.0), daxa_f32vec3(0.0, 0.0, 0.0));
    daxa_f32vec3 planet_zero = daxa_f32vec3(0.0, 0.0, 0.0);
    daxa_f32 planet_intersection_distance = ray_sphere_intersect_nearest(
        world_position, world_direction, planet_zero, deref(gpu_input).sky_settings.atmosphere_bottom);
    daxa_f32 atmosphere_intersection_distance = ray_sphere_intersect_nearest(
        world_position, world_direction, planet_zero, deref(gpu_input).sky_settings.atmosphere_top);

    daxa_f32 integration_length;
    /* ============================= CALCULATE INTERSECTIONS ============================ */
    if ((planet_intersection_distance == -1.0) && (atmosphere_intersection_distance == -1.0)) {
        /* ray does not intersect planet or atmosphere -> no point in raymarching*/
        return result;
    } else if ((planet_intersection_distance == -1.0) && (atmosphere_intersection_distance > 0.0)) {
        /* ray intersects only atmosphere */
        integration_length = atmosphere_intersection_distance;
    } else if ((planet_intersection_distance > 0.0) && (atmosphere_intersection_distance == -1.0)) {
        /* ray intersects only planet */
        integration_length = planet_intersection_distance;
    } else {
        /* ray intersects both planet and atmosphere -> return the first intersection */
        integration_length = min(planet_intersection_distance, atmosphere_intersection_distance);
    }
    daxa_f32 integration_step = integration_length / daxa_f32(sample_count);

    /* stores accumulated transmittance during the raymarch process */
    daxa_f32vec3 accum_transmittance = daxa_f32vec3(1.0, 1.0, 1.0);
    /* stores accumulated light contribution during the raymarch process */
    daxa_f32vec3 accum_light = daxa_f32vec3(0.0, 0.0, 0.0);
    daxa_f32 old_ray_shift = 0;

    /* ============================= RAYMARCH ==========================================  */
    for (daxa_i32 i = 0; i < sample_count; i++) {
        /* Sampling at 1/3rd of the integration step gives better results for exponential
           functions */
        daxa_f32 new_ray_shift = integration_length * (daxa_f32(i) + 0.3) / sample_count;
        integration_step = new_ray_shift - old_ray_shift;
        daxa_f32vec3 new_position = world_position + new_ray_shift * world_direction;
        old_ray_shift = new_ray_shift;

        /* Raymarch shifts the angle to the sun a bit recalculate */
        daxa_f32vec3 up_vector = normalize(new_position);
        TransmittanceParams transmittance_lut_params = TransmittanceParams(length(new_position), dot(sun_direction, up_vector));

        /* uv coordinates later used to sample transmittance texture */
        daxa_f32vec2 trans_texture_uv = transmittance_lut_to_uv(transmittance_lut_params, deref(gpu_input).sky_settings.atmosphere_bottom, deref(gpu_input).sky_settings.atmosphere_top);

        daxa_f32vec3 transmittance_to_sun = texture(daxa_sampler2D(transmittance_lut, deref(gpu_input).sampler_llc), trans_texture_uv).rgb;

        daxa_f32vec3 medium_scattering = sample_medium_scattering(gpu_input, new_position);
        daxa_f32vec3 medium_extinction = sample_medium_extinction(gpu_input, new_position);

        /* TODO: This probably should be a texture lookup altho might be slow*/
        daxa_f32vec3 trans_increase_over_integration_step = exp(-(medium_extinction * integration_step));
        /* Check if current position is in earth's shadow */
        daxa_f32 earth_intersection_distance = ray_sphere_intersect_nearest(
            new_position, sun_direction, planet_zero + PLANET_RADIUS_OFFSET * up_vector, deref(gpu_input).sky_settings.atmosphere_bottom);
        daxa_f32 in_earth_shadow = earth_intersection_distance == -1.0 ? 1.0 : 0.0;

        /* Light arriving from the sun to this point */
        daxa_f32vec3 sunLight = in_earth_shadow * transmittance_to_sun * medium_scattering * uniformPhase;
        daxa_f32vec3 multiscattered_cont_int = (medium_scattering - medium_scattering * trans_increase_over_integration_step) / medium_extinction;
        daxa_f32vec3 inscatteredContInt = (sunLight - sunLight * trans_increase_over_integration_step) / medium_extinction;

        if (medium_extinction.r == 0.0) {
            multiscattered_cont_int.r = 0.0;
            inscatteredContInt.r = 0.0;
        }
        if (medium_extinction.g == 0.0) {
            multiscattered_cont_int.g = 0.0;
            inscatteredContInt.g = 0.0;
        }
        if (medium_extinction.b == 0.0) {
            multiscattered_cont_int.b = 0.0;
            inscatteredContInt.b = 0.0;
        }

        result.multiscattering += accum_transmittance * multiscattered_cont_int;
        accum_light += accum_transmittance * inscatteredContInt;
        // accum_light = accum_transmittance;
        accum_transmittance *= trans_increase_over_integration_step;
    }
    result.luminance = accum_light;
    return result;
    /* TODO: Check for bounced light off the earth */
}

void main() {
    const daxa_f32 sample_count = 20;

    daxa_f32vec2 uv = (daxa_f32vec2(gl_GlobalInvocationID.xy) + daxa_f32vec2(0.5, 0.5)) /
                      SKY_MULTISCATTERING_RES;
    uv = daxa_f32vec2(from_subuv_to_unit(uv.x, SKY_MULTISCATTERING_RES.x),
                      from_subuv_to_unit(uv.y, SKY_MULTISCATTERING_RES.y));

    /* Mapping uv to multiscattering LUT parameters
       TODO -> Is the range from 0.0 to -1.0 really needed? */
    daxa_f32 sun_cos_zenith_angle = uv.x * 2.0 - 1.0;
    daxa_f32vec3 sun_direction = daxa_f32vec3(
        0.0,
        safe_sqrt(clamp(1.0 - sun_cos_zenith_angle * sun_cos_zenith_angle, 0.0, 1.0)),
        sun_cos_zenith_angle);

    daxa_f32 view_height = deref(gpu_input).sky_settings.atmosphere_bottom +
                           clamp(uv.y + PLANET_RADIUS_OFFSET, 0.0, 1.0) *
                               (deref(gpu_input).sky_settings.atmosphere_top - deref(gpu_input).sky_settings.atmosphere_bottom - PLANET_RADIUS_OFFSET);

    daxa_f32vec3 world_position = daxa_f32vec3(0.0, 0.0, view_height);

    daxa_f32 sample_idx = gl_LocalInvocationID.z;
    // local thread dependent raymarch
    {
#define USE_HILL_SAMPLING 0
#if USE_HILL_SAMPLING
#define SQRTSAMPLECOUNT 8
        const daxa_f32 sqrt_sample = daxa_f32(SQRTSAMPLECOUNT);
        daxa_f32 i = 0.5 + daxa_f32(sample_idx / SQRTSAMPLECOUNT);
        daxa_f32 j = 0.5 + mod(sample_idx, SQRTSAMPLECOUNT);
        daxa_f32 randA = i / sqrt_sample;
        daxa_f32 randB = j / sqrt_sample;

        daxa_f32 theta = 2.0 * M_PI * randA;
        daxa_f32 phi = M_PI * randB;
#else
        /* Fibbonaci lattice -> http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/ */
        daxa_f32 theta = acos(1.0 - 2.0 * (sample_idx + 0.5) / SPHERE_SAMPLES);
        daxa_f32 phi = (2 * M_PI * sample_idx) / GOLDEN_RATIO;
#endif

        daxa_f32vec3 world_direction = daxa_f32vec3(cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi));
        RaymarchResult result = integrate_scattered_luminance(world_position, world_direction, sun_direction, sample_count);

        multiscatt_shared[gl_LocalInvocationID.z] = result.multiscattering / SPHERE_SAMPLES;
        luminance_shared[gl_LocalInvocationID.z] = result.luminance / SPHERE_SAMPLES;
    }

    groupMemoryBarrier();
    barrier();

    if (gl_LocalInvocationID.z < 32) {
        multiscatt_shared[gl_LocalInvocationID.z] += multiscatt_shared[gl_LocalInvocationID.z + 32];
        luminance_shared[gl_LocalInvocationID.z] += luminance_shared[gl_LocalInvocationID.z + 32];
    }
    groupMemoryBarrier();
    barrier();
    if (gl_LocalInvocationID.z < 16) {
        multiscatt_shared[gl_LocalInvocationID.z] += multiscatt_shared[gl_LocalInvocationID.z + 16];
        luminance_shared[gl_LocalInvocationID.z] += luminance_shared[gl_LocalInvocationID.z + 16];
    }
    groupMemoryBarrier();
    barrier();
    if (gl_LocalInvocationID.z < 8) {
        multiscatt_shared[gl_LocalInvocationID.z] += multiscatt_shared[gl_LocalInvocationID.z + 8];
        luminance_shared[gl_LocalInvocationID.z] += luminance_shared[gl_LocalInvocationID.z + 8];
    }
    groupMemoryBarrier();
    barrier();
    if (gl_LocalInvocationID.z < 4) {
        multiscatt_shared[gl_LocalInvocationID.z] += multiscatt_shared[gl_LocalInvocationID.z + 4];
        luminance_shared[gl_LocalInvocationID.z] += luminance_shared[gl_LocalInvocationID.z + 4];
    }
    groupMemoryBarrier();
    barrier();
    if (gl_LocalInvocationID.z < 2) {
        multiscatt_shared[gl_LocalInvocationID.z] += multiscatt_shared[gl_LocalInvocationID.z + 2];
        luminance_shared[gl_LocalInvocationID.z] += luminance_shared[gl_LocalInvocationID.z + 2];
    }
    groupMemoryBarrier();
    barrier();
    if (gl_LocalInvocationID.z < 1) {
        multiscatt_shared[gl_LocalInvocationID.z] += multiscatt_shared[gl_LocalInvocationID.z + 1];
        luminance_shared[gl_LocalInvocationID.z] += luminance_shared[gl_LocalInvocationID.z + 1];
    }
    groupMemoryBarrier();
    barrier();
    if (gl_LocalInvocationID.z != 0)
        return;

    daxa_f32vec3 multiscatt_sum = multiscatt_shared[0];
    daxa_f32vec3 inscattered_luminance_sum = luminance_shared[0];

    const daxa_f32vec3 r = multiscatt_sum;
    const daxa_f32vec3 sum_of_all_multiscattering_events_contribution = daxa_f32vec3(1.0 / (1.0 - r.x), 1.0 / (1.0 - r.y), 1.0 / (1.0 - r.z));
    daxa_f32vec3 lum = inscattered_luminance_sum * sum_of_all_multiscattering_events_contribution;

    imageStore(daxa_image2D(multiscattering_lut), daxa_i32vec2(gl_GlobalInvocationID.xy), daxa_f32vec4(lum, 1.0));
}

#endif

#if SkySkyComputeShader

DAXA_DECL_PUSH_CONSTANT(SkySkyComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewIndex transmittance_lut = push.uses.transmittance_lut;
daxa_ImageViewIndex multiscattering_lut = push.uses.multiscattering_lut;
daxa_ImageViewIndex sky_lut = push.uses.sky_lut;

layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;
/* ============================= PHASE FUNCTIONS ============================ */
daxa_f32 cornette_shanks_mie_phase_function(daxa_f32 g, daxa_f32 cos_theta) {
    daxa_f32 k = 3.0 / (8.0 * M_PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cos_theta * cos_theta) / pow(1.0 + g * g - 2.0 * g * -cos_theta, 1.5);
}

float kleinNishinaPhase(float cosTheta, float e) {
    return e / (2.0 * M_PI * (e * (1.0 - cosTheta) + 1.0) * log(2.0 * e + 1.0));
}

daxa_f32 rayleigh_phase(daxa_f32 cos_theta) {
    daxa_f32 factor = 3.0 / (16.0 * M_PI);
    return factor * (1.0 + cos_theta * cos_theta);
}
/* ========================================================================== */

daxa_f32vec3 get_multiple_scattering(daxa_f32vec3 world_position, daxa_f32 view_zenith_cos_angle) {
    daxa_f32vec2 uv = clamp(daxa_f32vec2(
                                view_zenith_cos_angle * 0.5 + 0.5,
                                (length(world_position) - deref(gpu_input).sky_settings.atmosphere_bottom) /
                                    (deref(gpu_input).sky_settings.atmosphere_top - deref(gpu_input).sky_settings.atmosphere_bottom)),
                            0.0, 1.0);
    uv = daxa_f32vec2(from_unit_to_subuv(uv.x, SKY_MULTISCATTERING_RES.x),
                      from_unit_to_subuv(uv.y, SKY_MULTISCATTERING_RES.y));

    return texture(daxa_sampler2D(multiscattering_lut, deref(gpu_input).sampler_llc), uv).rgb;
}

daxa_f32vec3 integrate_scattered_luminance(daxa_f32vec3 world_position,
                                           daxa_f32vec3 world_direction, daxa_f32vec3 sun_direction, daxa_i32 sample_count) {
    daxa_f32vec3 planet_zero = daxa_f32vec3(0.0, 0.0, 0.0);
    daxa_f32 planet_intersection_distance = ray_sphere_intersect_nearest(
        world_position, world_direction, planet_zero, deref(gpu_input).sky_settings.atmosphere_bottom);
    daxa_f32 atmosphere_intersection_distance = ray_sphere_intersect_nearest(
        world_position, world_direction, planet_zero, deref(gpu_input).sky_settings.atmosphere_top);

    daxa_f32 integration_length;
    /* ============================= CALCULATE INTERSECTIONS ============================ */
    if ((planet_intersection_distance == -1.0) && (atmosphere_intersection_distance == -1.0)) {
        /* ray does not intersect planet or atmosphere -> no point in raymarching*/
        return daxa_f32vec3(0.0, 0.0, 0.0);
    } else if ((planet_intersection_distance == -1.0) && (atmosphere_intersection_distance > 0.0)) {
        /* ray intersects only atmosphere */
        integration_length = atmosphere_intersection_distance;
    } else if ((planet_intersection_distance > 0.0) && (atmosphere_intersection_distance == -1.0)) {
        /* ray intersects only planet */
        integration_length = planet_intersection_distance;
    } else {
        /* ray intersects both planet and atmosphere -> return the first intersection */
        integration_length = min(planet_intersection_distance, atmosphere_intersection_distance);
    }

    daxa_f32 cos_theta = dot(sun_direction, world_direction);
    daxa_f32 mie_phase_value = kleinNishinaPhase(cos_theta, 2800.0);
    daxa_f32 rayleigh_phase_value = rayleigh_phase(cos_theta);

    daxa_f32vec3 accum_transmittance = daxa_f32vec3(1.0, 1.0, 1.0);
    daxa_f32vec3 accum_light = daxa_f32vec3(0.0, 0.0, 0.0);
    /* ============================= RAYMARCH ============================ */
    for (daxa_i32 i = 0; i < sample_count; i++) {
        /* Step size computation */
        daxa_f32 step_0 = daxa_f32(i) / sample_count;
        daxa_f32 step_1 = daxa_f32(i + 1) / sample_count;

        /* Nonuniform step size*/
        step_0 *= step_0;
        step_1 *= step_1;

        step_0 = step_0 * integration_length;
        step_1 = step_1 > 1.0 ? integration_length : step_1 * integration_length;
        /* Sample at one third of the integrated interval -> better results for exponential functions */
        daxa_f32 integration_step = step_0 + (step_1 - step_0) * 0.3;
        daxa_f32 d_int_step = step_1 - step_0;

        /* Position shift */
        daxa_f32vec3 new_position = world_position + integration_step * world_direction;
        ScatteringSample medium_scattering = sample_medium_scattering_detailed(gpu_input, new_position);
        daxa_f32vec3 medium_extinction = sample_medium_extinction(gpu_input, new_position);

        daxa_f32vec3 up_vector = normalize(new_position);
        TransmittanceParams transmittance_lut_params = TransmittanceParams(length(new_position), dot(sun_direction, up_vector));

        /* uv coordinates later used to sample transmittance texture */
        daxa_f32vec2 trans_texture_uv = transmittance_lut_to_uv(transmittance_lut_params, deref(gpu_input).sky_settings.atmosphere_bottom, deref(gpu_input).sky_settings.atmosphere_top);
        daxa_f32vec3 transmittance_to_sun = texture(daxa_sampler2D(transmittance_lut, deref(gpu_input).sampler_llc), trans_texture_uv).rgb;

        daxa_f32vec3 phase_times_scattering = medium_scattering.mie * mie_phase_value + medium_scattering.ray * rayleigh_phase_value;

        daxa_f32 earth_intersection_distance = ray_sphere_intersect_nearest(
            new_position, sun_direction, planet_zero, deref(gpu_input).sky_settings.atmosphere_bottom);
        daxa_f32 in_earth_shadow = earth_intersection_distance == -1.0 ? 1.0 : 0.0;

        daxa_f32vec3 multiscattered_luminance = get_multiple_scattering(new_position, dot(sun_direction, up_vector));

        /* Light arriving from the sun to this point */
        daxa_f32vec3 sun_light = in_earth_shadow * transmittance_to_sun * phase_times_scattering +
                                 multiscattered_luminance * (medium_scattering.ray + medium_scattering.mie);

        /* TODO: This probably should be a texture lookup*/
        daxa_f32vec3 trans_increase_over_integration_step = exp(-(medium_extinction * d_int_step));

        daxa_f32vec3 sun_light_integ = (sun_light - sun_light * trans_increase_over_integration_step) / medium_extinction;

        if (medium_extinction.r == 0.0) {
            sun_light_integ.r = 0.0;
        }
        if (medium_extinction.g == 0.0) {
            sun_light_integ.g = 0.0;
        }
        if (medium_extinction.b == 0.0) {
            sun_light_integ.b = 0.0;
        }

        accum_light += accum_transmittance * sun_light_integ;
        accum_transmittance *= trans_increase_over_integration_step;
    }
    return accum_light;
}

void main() {
    if (any(greaterThan(gl_GlobalInvocationID.xy, SKY_SKY_RES))) {
        return;
    }

    // Hardcode player position to be 100 meters above sea level
    daxa_f32vec3 world_position = daxa_f32vec3(0.0, 0.0, deref(gpu_input).sky_settings.atmosphere_bottom + ATMOSPHERE_CAMERA_HEIGHT);

    daxa_f32vec2 uv = daxa_f32vec2(gl_GlobalInvocationID.xy) / daxa_f32vec2(SKY_SKY_RES);
    SkyviewParams skyview_params = uv_to_skyview_lut_params(
        uv,
        deref(gpu_input).sky_settings.atmosphere_bottom,
        deref(gpu_input).sky_settings.atmosphere_top,
        SKY_SKY_RES,
        length(world_position));

    daxa_f32 sun_zenith_cos_angle = dot(normalize(world_position), deref(gpu_input).sky_settings.sun_direction);
    // sin^2 + cos^2 = 1 -> sqrt(1 - cos^2) = sin
    // rotate the sun direction so that we are aligned with the y = 0 axis
    daxa_f32vec3 local_sun_direction = normalize(daxa_f32vec3(
        safe_sqrt(1.0 - sun_zenith_cos_angle * sun_zenith_cos_angle),
        0.0,
        sun_zenith_cos_angle));

    daxa_f32vec3 world_direction = daxa_f32vec3(
        cos(skyview_params.light_view_angle) * sin(skyview_params.view_zenith_angle),
        sin(skyview_params.light_view_angle) * sin(skyview_params.view_zenith_angle),
        cos(skyview_params.view_zenith_angle));

    if (!move_to_top_atmosphere(world_position, world_direction, deref(gpu_input).sky_settings.atmosphere_bottom, deref(gpu_input).sky_settings.atmosphere_top)) {
        /* No intersection with the atmosphere */
        imageStore(daxa_image2D(sky_lut), daxa_i32vec2(gl_GlobalInvocationID.xy), daxa_f32vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }
    daxa_f32vec3 luminance = integrate_scattered_luminance(world_position, world_direction, local_sun_direction, 30);
    imageStore(daxa_image2D(sky_lut), daxa_i32vec2(gl_GlobalInvocationID.xy), daxa_f32vec4(luminance, 1.0));
}

#endif

#if SkyCubeComputeShader

DAXA_DECL_PUSH_CONSTANT(SkyCubeComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewIndex transmittance_lut = push.uses.transmittance_lut;
daxa_ImageViewIndex sky_lut = push.uses.sky_lut;
daxa_ImageViewIndex sky_cube = push.uses.sky_cube;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    daxa_u32vec3 px = gl_GlobalInvocationID.xyz;
    uint face = px.z;
    vec2 uv = (px.xy + 0.5) / SKY_CUBE_RES;

    vec3 output_dir = normalize(CUBE_MAP_FACE_ROTATION(face) * vec3(uv * 2 - 1, -1.0));
    const mat3 basis = build_orthonormal_basis(output_dir);

    AtmosphereLightingInfo sky_lighting = get_atmosphere_lighting(gpu_input, sky_lut, transmittance_lut, output_dir, output_dir);
    vec4 result = vec4(sky_lighting.atmosphere_direct_illuminance, 1);

    imageStore(daxa_image2DArray(sky_cube), ivec3(px), result);
}

#endif

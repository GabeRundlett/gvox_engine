#include <shared/app.inl>

#include <utils/math.glsl>
#include <utils/sky.glsl>
const f32 PLANET_RADIUS_OFFSET = 0.01;

struct TransmittanceParams {
    f32 height;
    f32 zenith_cos_angle;
};

///	Transmittance LUT uses not uniform mapping -> transfer from mapping to texture uv
///	@param parameters
/// @param atmosphere_bottom - bottom radius of the atmosphere in km
/// @param atmosphere_top - top radius of the atmosphere in km
///	@return - uv of the corresponding texel
f32vec2 transmittance_lut_to_uv(TransmittanceParams parameters, f32 atmosphere_bottom, f32 atmosphere_top) {
    f32 H = safe_sqrt(atmosphere_top * atmosphere_top - atmosphere_bottom * atmosphere_bottom);
    f32 rho = safe_sqrt(parameters.height * parameters.height - atmosphere_bottom * atmosphere_bottom);

    f32 discriminant = parameters.height * parameters.height *
                           (parameters.zenith_cos_angle * parameters.zenith_cos_angle - 1.0) +
                       atmosphere_top * atmosphere_top;
    /* Distance to top atmosphere boundary */
    f32 d = max(0.0, (-parameters.height * parameters.zenith_cos_angle + safe_sqrt(discriminant)));

    f32 d_min = atmosphere_top - parameters.height;
    f32 d_max = rho + H;
    f32 mu = (d - d_min) / (d_max - d_min);
    f32 r = rho / H;

    return f32vec2(mu, r);
}

/// Transmittance LUT uses not uniform mapping -> transfer from uv to this mapping
/// @param uv - uv in the range [0,1]
/// @param atmosphere_bottom - bottom radius of the atmosphere in km
/// @param atmosphere_top - top radius of the atmosphere in km
/// @return - TransmittanceParams structure
TransmittanceParams uv_to_transmittance_lut_params(f32vec2 uv, f32 atmosphere_bottom, f32 atmosphere_top) {
    TransmittanceParams params;
    f32 H = safe_sqrt(atmosphere_top * atmosphere_top - atmosphere_bottom * atmosphere_bottom.x);

    f32 rho = H * uv.y;
    params.height = safe_sqrt(rho * rho + atmosphere_bottom * atmosphere_bottom);

    f32 d_min = atmosphere_top - params.height;
    f32 d_max = rho + H;
    f32 d = d_min + uv.x * (d_max - d_min);

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
SkyviewParams uv_to_skyview_lut_params(f32vec2 uv, f32 atmosphere_bottom,
                                       f32 atmosphere_top, f32vec2 skyview_dimensions, f32 view_height) {
    /* Constrain uvs to valid sub texel range
    (avoid zenith derivative issue making LUT usage visible) */
    uv = f32vec2(from_subuv_to_unit(uv.x, skyview_dimensions.x),
                 from_subuv_to_unit(uv.y, skyview_dimensions.y));

    f32 beta = asin(atmosphere_bottom / view_height);
    f32 zenith_horizon_angle = PI - beta;

    f32 view_zenith_angle;
    f32 light_view_angle;
    /* Nonuniform mapping near the horizon to avoid artefacts */
    if (uv.y < 0.5) {
        f32 coord = 1.0 - (1.0 - 2.0 * uv.y) * (1.0 - 2.0 * uv.y);
        view_zenith_angle = zenith_horizon_angle * coord;
    } else {
        f32 coord = (uv.y * 2.0 - 1.0) * (uv.y * 2.0 - 1.0);
        view_zenith_angle = zenith_horizon_angle + beta * coord;
    }
    light_view_angle = (uv.x * uv.x) * PI;
    return SkyviewParams(view_zenith_angle, light_view_angle);
}

/// Moves to the nearest intersection with top of the atmosphere in the direction specified in
/// world_direction
/// @param world_position - current world position -> will be changed to new pos at the top of
/// 		the atmosphere if there exists such intersection
/// @param world_direction - the direction in which the shift will be done
/// @param atmosphere_bottom - bottom of the atmosphere in km
/// @param atmosphere_top - top of the atmosphere in km
b32 move_to_top_atmosphere(inout f32vec3 world_position, f32vec3 world_direction,
                           f32 atmosphere_bottom, f32 atmosphere_top) {
    f32vec3 planet_origin = f32vec3(0.0, 0.0, 0.0);
    /* Check if the world_position is outside of the atmosphere */
    if (length(world_position) > atmosphere_top) {
        f32 dist_to_top_atmo_intersection = ray_sphere_intersect_nearest(
            world_position, world_direction, planet_origin, atmosphere_top);

        /* No intersection with the atmosphere */
        if (dist_to_top_atmo_intersection == -1.0) {
            return false;
        } else {
            f32vec3 up_offset = normalize(world_position) * -PLANET_RADIUS_OFFSET;
            world_position += world_direction * dist_to_top_atmo_intersection + up_offset;
        }
    }
    /* Position is in or at the top of the atmosphere */
    return true;
}

/// @param params - buffer reference to the atmosphere parameters buffer
/// @param position - position in the world where the sample is to be taken
/// @return atmosphere extinction at the desired point
f32vec3 sample_medium_extinction(daxa_BufferPtr(GpuInput) gpu_input, f32vec3 position) {
    const f32 height = length(position) - deref(gpu_input).sky_settings.atmosphere_bottom;

    const f32 density_mie = exp(deref(gpu_input).sky_settings.mie_density[1].exp_scale * height);
    const f32 density_ray = exp(deref(gpu_input).sky_settings.rayleigh_density[1].exp_scale * height);
    // const f32 density_ozo = clamp(height < deref(gpu_input).sky_settings.absorption_density[0].layer_width ?
    //     deref(gpu_input).sky_settings.absorption_density[0].lin_term * height + deref(gpu_input).sky_settings.absorption_density[0].const_term :
    //     deref(gpu_input).sky_settings.absorption_density[1].lin_term * height + deref(gpu_input).sky_settings.absorption_density[1].const_term,
    //     0.0, 1.0);
    const f32 density_ozo = exp(-max(0.0, 35.0 - height) * (1.0 / 5.0)) * exp(-max(0.0, height - 35.0) * (1.0 / 15.0)) * 2;
    f32vec3 mie_extinction = deref(gpu_input).sky_settings.mie_extinction * density_mie;
    f32vec3 ray_extinction = deref(gpu_input).sky_settings.rayleigh_scattering * density_ray;
    f32vec3 ozo_extinction = deref(gpu_input).sky_settings.absorption_extinction * density_ozo;

    return mie_extinction + ray_extinction + ozo_extinction;
}

/// @param params - buffer reference to the atmosphere parameters buffer
/// @param position - position in the world where the sample is to be taken
/// @return atmosphere scattering at the desired point
f32vec3 sample_medium_scattering(daxa_BufferPtr(GpuInput) gpu_input, f32vec3 position) {
    const f32 height = length(position) - deref(gpu_input).sky_settings.atmosphere_bottom;

    const f32 density_mie = exp(deref(gpu_input).sky_settings.mie_density[1].exp_scale * height);
    const f32 density_ray = exp(deref(gpu_input).sky_settings.rayleigh_density[1].exp_scale * height);

    f32vec3 mie_scattering = deref(gpu_input).sky_settings.mie_scattering * density_mie;
    f32vec3 ray_scattering = deref(gpu_input).sky_settings.rayleigh_scattering * density_ray;
    /* Not considering ozon scattering in current version of this model */
    f32vec3 ozo_scattering = f32vec3(0.0, 0.0, 0.0);

    return mie_scattering + ray_scattering + ozo_scattering;
}

struct ScatteringSample {
    f32vec3 mie;
    f32vec3 ray;
};
/// @param params - buffer reference to the atmosphere parameters buffer
/// @param position - position in the world where the sample is to be taken
/// @return Scattering sample struct
// TODO(msakmary) Fix this!!
ScatteringSample sample_medium_scattering_detailed(daxa_BufferPtr(GpuInput) gpu_input, f32vec3 position) {
    const f32 height = length(position) - deref(gpu_input).sky_settings.atmosphere_bottom;

    const f32 density_mie = exp(deref(gpu_input).sky_settings.mie_density[1].exp_scale * height);
    const f32 density_ray = exp(deref(gpu_input).sky_settings.rayleigh_density[1].exp_scale * height);
    const f32 density_ozo = clamp(height < deref(gpu_input).sky_settings.absorption_density[0].layer_width ? deref(gpu_input).sky_settings.absorption_density[0].lin_term * height + deref(gpu_input).sky_settings.absorption_density[0].const_term : deref(gpu_input).sky_settings.absorption_density[1].lin_term * height + deref(gpu_input).sky_settings.absorption_density[1].const_term,
                                  0.0, 1.0);

    f32vec3 mie_scattering = deref(gpu_input).sky_settings.mie_scattering * density_mie;
    f32vec3 ray_scattering = deref(gpu_input).sky_settings.rayleigh_scattering * density_ray;
    /* Not considering ozon scattering in current version of this model */
    f32vec3 ozo_scattering = f32vec3(0.0, 0.0, 0.0);

    return ScatteringSample(mie_scattering, ray_scattering);
}

#if SKY_TRANSMITTANCE_COMPUTE

layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;
f32vec3 integrate_transmittance(f32vec3 world_position, f32vec3 world_direction, u32 sample_count) {
    /* The length of ray between position and nearest atmosphere top boundary */
    f32 integration_length = ray_sphere_intersect_nearest(
        world_position,
        world_direction,
        f32vec3(0.0, 0.0, 0.0),
        deref(gpu_input).sky_settings.atmosphere_top);

    f32 integration_step = integration_length / f32(sample_count);

    /* Result of the integration */
    f32vec3 optical_depth = f32vec3(0.0, 0.0, 0.0);

    for (i32 i = 0; i < sample_count; i++) {
        /* Move along the world direction ray to new position */
        f32vec3 new_pos = world_position + i * integration_step * world_direction;
        f32vec3 atmosphere_extinction = sample_medium_extinction(gpu_input, new_pos);
        optical_depth += atmosphere_extinction * integration_step;
    }
    return optical_depth;
}

void main() {
    if (any(greaterThan(gl_GlobalInvocationID.xy, SKY_TRANSMITTANCE_RES))) {
        return;
    }

    f32vec2 uv = f32vec2(gl_GlobalInvocationID.xy) / f32vec2(SKY_TRANSMITTANCE_RES);

    TransmittanceParams mapping = uv_to_transmittance_lut_params(
        uv,
        deref(gpu_input).sky_settings.atmosphere_bottom,
        deref(gpu_input).sky_settings.atmosphere_top);

    f32vec3 world_position = f32vec3(0.0, 0.0, mapping.height);
    f32vec3 world_direction = f32vec3(
        safe_sqrt(1.0 - mapping.zenith_cos_angle * mapping.zenith_cos_angle),
        0.0,
        mapping.zenith_cos_angle);

    f32vec3 transmittance = exp(-integrate_transmittance(world_position, world_direction, 400));

    imageStore(daxa_image2D(transmittance_lut), i32vec2(gl_GlobalInvocationID.xy), f32vec4(transmittance, 1.0));
}

#endif

#if SKY_MULTISCATTERING_COMPUTE

layout(local_size_x = 1, local_size_y = 1, local_size_z = 64) in;
/* This number should match the number of local threads -> z dimension */
const f32 SPHERE_SAMPLES = 64.0;
const f32 GOLDEN_RATIO = 1.6180339;
const f32 uniformPhase = 1.0 / (4.0 * PI);

shared f32vec3 multiscatt_shared[64];
shared f32vec3 luminance_shared[64];

struct RaymarchResult 
{
    f32vec3 luminance;
    f32vec3 multiscattering;
};

RaymarchResult integrate_scattered_luminance(f32vec3 world_position, f32vec3 world_direction, f32vec3 sun_direction, f32 sample_count)
{
    RaymarchResult result = RaymarchResult(f32vec3(0.0, 0.0, 0.0), f32vec3(0.0, 0.0, 0.0));
    f32vec3 planet_zero = f32vec3(0.0, 0.0, 0.0);
    f32 planet_intersection_distance = ray_sphere_intersect_nearest(
        world_position, world_direction, planet_zero, deref(gpu_input).sky_settings.atmosphere_bottom);
    f32 atmosphere_intersection_distance = ray_sphere_intersect_nearest(
        world_position, world_direction, planet_zero, deref(gpu_input).sky_settings.atmosphere_top);
    
    f32 integration_length;
    /* ============================= CALCULATE INTERSECTIONS ============================ */
    if((planet_intersection_distance == -1.0) && (atmosphere_intersection_distance == -1.0)){
        /* ray does not intersect planet or atmosphere -> no point in raymarching*/
        return result;
    } 
    else if((planet_intersection_distance == -1.0) && (atmosphere_intersection_distance > 0.0)){
        /* ray intersects only atmosphere */
        integration_length = atmosphere_intersection_distance;
    }
    else if((planet_intersection_distance > 0.0) && (atmosphere_intersection_distance == -1.0)){
        /* ray intersects only planet */
        integration_length = planet_intersection_distance;
    } else {
        /* ray intersects both planet and atmosphere -> return the first intersection */
        integration_length = min(planet_intersection_distance, atmosphere_intersection_distance);
    }
    f32 integration_step = integration_length / f32(sample_count);

    /* stores accumulated transmittance during the raymarch process */
    f32vec3 accum_transmittance = f32vec3(1.0, 1.0, 1.0);
    /* stores accumulated light contribution during the raymarch process */
    f32vec3 accum_light = f32vec3(0.0, 0.0, 0.0);
    f32 old_ray_shift = 0;

    /* ============================= RAYMARCH ==========================================  */
    for(i32 i = 0; i < sample_count; i++)
    {
        /* Sampling at 1/3rd of the integration step gives better results for exponential
           functions */
        f32 new_ray_shift = integration_length * (f32(i) + 0.3) / sample_count;
        integration_step = new_ray_shift - old_ray_shift;
        f32vec3 new_position = world_position + new_ray_shift * world_direction;
        old_ray_shift = new_ray_shift;

        /* Raymarch shifts the angle to the sun a bit recalculate */
        f32vec3 up_vector = normalize(new_position);
        TransmittanceParams transmittance_lut_params = TransmittanceParams(length(new_position), dot(sun_direction, up_vector));

        /* uv coordinates later used to sample transmittance texture */
        f32vec2 trans_texture_uv = transmittance_lut_to_uv(transmittance_lut_params, deref(gpu_input).sky_settings.atmosphere_bottom, deref(gpu_input).sky_settings.atmosphere_top);

        f32vec3 transmittance_to_sun = texture(daxa_sampler2D(transmittance_lut, deref(gpu_input).sampler_llc), trans_texture_uv).rgb;

        f32vec3 medium_scattering = sample_medium_scattering(gpu_input, new_position);
        f32vec3 medium_extinction = sample_medium_extinction(gpu_input, new_position);

        /* TODO: This probably should be a texture lookup altho might be slow*/
        f32vec3 trans_increase_over_integration_step = exp(-(medium_extinction * integration_step));
        /* Check if current position is in earth's shadow */
        f32 earth_intersection_distance = ray_sphere_intersect_nearest(
            new_position, sun_direction, planet_zero + PLANET_RADIUS_OFFSET * up_vector, deref(gpu_input).sky_settings.atmosphere_bottom);
        f32 in_earth_shadow = earth_intersection_distance == -1.0 ? 1.0 : 0.0;

        /* Light arriving from the sun to this point */
        f32vec3 sunLight = in_earth_shadow * transmittance_to_sun * medium_scattering * uniformPhase;
        f32vec3 multiscattered_cont_int = (medium_scattering - medium_scattering * trans_increase_over_integration_step) / medium_extinction;
        f32vec3 inscatteredContInt = (sunLight - sunLight * trans_increase_over_integration_step) / medium_extinction;

        if(medium_extinction.r == 0.0) { multiscattered_cont_int.r = 0.0; inscatteredContInt.r = 0.0; }
        if(medium_extinction.g == 0.0) { multiscattered_cont_int.g = 0.0; inscatteredContInt.g = 0.0; }
        if(medium_extinction.b == 0.0) { multiscattered_cont_int.b = 0.0; inscatteredContInt.b = 0.0; }

        result.multiscattering += accum_transmittance * multiscattered_cont_int;
        accum_light += accum_transmittance * inscatteredContInt;
        // accum_light = accum_transmittance;
        accum_transmittance *= trans_increase_over_integration_step;
    }
    result.luminance = accum_light;
    return result;
    /* TODO: Check for bounced light off the earth */
}

void main()
{
    const f32 sample_count = 20;

    f32vec2 uv = (f32vec2(gl_GlobalInvocationID.xy) + f32vec2(0.5, 0.5)) / 
                  SKY_MULTISCATTERING_RES;
    uv = f32vec2(from_subuv_to_unit(uv.x, SKY_MULTISCATTERING_RES.x),
                 from_subuv_to_unit(uv.y, SKY_MULTISCATTERING_RES.y));
    
    /* Mapping uv to multiscattering LUT parameters
       TODO -> Is the range from 0.0 to -1.0 really needed? */
    f32 sun_cos_zenith_angle = uv.x * 2.0 - 1.0;
    f32vec3 sun_direction = f32vec3(
        0.0,
        safe_sqrt(clamp(1.0 - sun_cos_zenith_angle * sun_cos_zenith_angle, 0.0, 1.0)),
        sun_cos_zenith_angle
    );

   f32 view_height = deref(gpu_input).sky_settings.atmosphere_bottom + 
        clamp(uv.y + PLANET_RADIUS_OFFSET, 0.0, 1.0) *
        (deref(gpu_input).sky_settings.atmosphere_top - deref(gpu_input).sky_settings.atmosphere_bottom - PLANET_RADIUS_OFFSET);

    f32vec3 world_position = f32vec3(0.0, 0.0, view_height);

    f32 sample_idx = gl_LocalInvocationID.z;
    // local thread dependent raymarch
    { 
        #define USE_HILL_SAMPLING 0
        #if USE_HILL_SAMPLING
            #define SQRTSAMPLECOUNT 8
            const f32 sqrt_sample = f32(SQRTSAMPLECOUNT);
            f32 i = 0.5 + f32(sample_idx / SQRTSAMPLECOUNT);
            f32 j = 0.5 + mod(sample_idx, SQRTSAMPLECOUNT);
            f32 randA = i / sqrt_sample;
            f32 randB = j / sqrt_sample;

            f32 theta = 2.0 * PI * randA;
            f32 phi = PI * randB;
        #else
        /* Fibbonaci lattice -> http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/ */
            f32 theta = acos( 1.0 - 2.0 * (sample_idx + 0.5) / SPHERE_SAMPLES );
            f32 phi = (2 * PI * sample_idx) / GOLDEN_RATIO;
        #endif


        f32vec3 world_direction = f32vec3( cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi));
        RaymarchResult result = integrate_scattered_luminance(world_position, world_direction, sun_direction, sample_count);

        multiscatt_shared[gl_LocalInvocationID.z] = result.multiscattering / SPHERE_SAMPLES;
        luminance_shared[gl_LocalInvocationID.z] = result.luminance / SPHERE_SAMPLES;
    }

    groupMemoryBarrier();
    barrier();

    if(gl_LocalInvocationID.z < 32)
    {
        multiscatt_shared[gl_LocalInvocationID.z] += multiscatt_shared[gl_LocalInvocationID.z + 32];
        luminance_shared[gl_LocalInvocationID.z] += luminance_shared[gl_LocalInvocationID.z + 32];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationID.z < 16)
    {
        multiscatt_shared[gl_LocalInvocationID.z] += multiscatt_shared[gl_LocalInvocationID.z + 16];
        luminance_shared[gl_LocalInvocationID.z] += luminance_shared[gl_LocalInvocationID.z + 16];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationID.z < 8)
    {
        multiscatt_shared[gl_LocalInvocationID.z] += multiscatt_shared[gl_LocalInvocationID.z + 8];
        luminance_shared[gl_LocalInvocationID.z] += luminance_shared[gl_LocalInvocationID.z + 8];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationID.z < 4)
    {
        multiscatt_shared[gl_LocalInvocationID.z] += multiscatt_shared[gl_LocalInvocationID.z + 4];
        luminance_shared[gl_LocalInvocationID.z] += luminance_shared[gl_LocalInvocationID.z + 4];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationID.z < 2)
    {
        multiscatt_shared[gl_LocalInvocationID.z] += multiscatt_shared[gl_LocalInvocationID.z + 2];
        luminance_shared[gl_LocalInvocationID.z] += luminance_shared[gl_LocalInvocationID.z + 2];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationID.z < 1)
    {
        multiscatt_shared[gl_LocalInvocationID.z] += multiscatt_shared[gl_LocalInvocationID.z + 1];
        luminance_shared[gl_LocalInvocationID.z] += luminance_shared[gl_LocalInvocationID.z + 1];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationID.z != 0)
        return;

    f32vec3 multiscatt_sum = multiscatt_shared[0];
    f32vec3 inscattered_luminance_sum = luminance_shared[0];

    const f32vec3 r = multiscatt_sum;
    const f32vec3 sum_of_all_multiscattering_events_contribution = f32vec3(1.0/ (1.0 -r.x),1.0/ (1.0 -r.y),1.0/ (1.0 -r.z));
    f32vec3 lum = inscattered_luminance_sum * sum_of_all_multiscattering_events_contribution;

    imageStore(daxa_image2D(multiscattering_lut), i32vec2(gl_GlobalInvocationID.xy), f32vec4(lum, 1.0));
}

#endif

#if SKY_SKY_COMPUTE

layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;
/* ============================= PHASE FUNCTIONS ============================ */
f32 cornette_shanks_mie_phase_function(f32 g, f32 cos_theta)
{
    f32 k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cos_theta * cos_theta) / pow(1.0 + g * g - 2.0 * g * -cos_theta, 1.5);
}

f32 rayleigh_phase(f32 cos_theta)
{
    f32 factor = 3.0 / (16.0 * PI);
    return factor * (1.0 + cos_theta * cos_theta);
}
/* ========================================================================== */

f32vec3 get_multiple_scattering(f32vec3 world_position, f32 view_zenith_cos_angle)
{
    f32vec2 uv = clamp(f32vec2( 
        view_zenith_cos_angle * 0.5 + 0.5,
        (length(world_position) - deref(gpu_input).sky_settings.atmosphere_bottom) /
        (deref(gpu_input).sky_settings.atmosphere_top - deref(gpu_input).sky_settings.atmosphere_bottom)),
        0.0, 1.0);
    uv = f32vec2(from_unit_to_subuv(uv.x, SKY_MULTISCATTERING_RES.x),
                 from_unit_to_subuv(uv.y, SKY_MULTISCATTERING_RES.y));

    return texture(daxa_sampler2D(multiscattering_lut, deref(gpu_input).sampler_llc), uv).rgb;
}

f32vec3 integrate_scattered_luminance(f32vec3 world_position, 
    f32vec3 world_direction, f32vec3 sun_direction, i32 sample_count)
{
    f32vec3 planet_zero = f32vec3(0.0, 0.0, 0.0);
    f32 planet_intersection_distance = ray_sphere_intersect_nearest(
        world_position, world_direction, planet_zero, deref(gpu_input).sky_settings.atmosphere_bottom);
    f32 atmosphere_intersection_distance = ray_sphere_intersect_nearest(
        world_position, world_direction, planet_zero, deref(gpu_input).sky_settings.atmosphere_top);
    
    f32 integration_length;
    /* ============================= CALCULATE INTERSECTIONS ============================ */
    if((planet_intersection_distance == -1.0) && (atmosphere_intersection_distance == -1.0)){
        /* ray does not intersect planet or atmosphere -> no point in raymarching*/
        return f32vec3(0.0, 0.0, 0.0);
    } 
    else if((planet_intersection_distance == -1.0) && (atmosphere_intersection_distance > 0.0)){
        /* ray intersects only atmosphere */
        integration_length = atmosphere_intersection_distance;
    }
    else if((planet_intersection_distance > 0.0) && (atmosphere_intersection_distance == -1.0)){
        /* ray intersects only planet */
        integration_length = planet_intersection_distance;
    } else {
        /* ray intersects both planet and atmosphere -> return the first intersection */
        integration_length = min(planet_intersection_distance, atmosphere_intersection_distance);
    }

    f32 cos_theta = dot(sun_direction, world_direction);
    f32 mie_phase_value = cornette_shanks_mie_phase_function(deref(gpu_input).sky_settings.mie_phase_function_g, -cos_theta);
    f32 rayleigh_phase_value = rayleigh_phase(cos_theta);

    f32vec3 accum_transmittance = f32vec3(1.0, 1.0, 1.0);
    f32vec3 accum_light = f32vec3(0.0, 0.0, 0.0);
    /* ============================= RAYMARCH ============================ */
    for(i32 i = 0; i < sample_count; i++)
    {
        /* Step size computation */
        f32 step_0 = f32(i) / sample_count;
        f32 step_1 = f32(i + 1) / sample_count;

        /* Nonuniform step size*/
        step_0 *= step_0;
        step_1 *= step_1;

        step_0 = step_0 * integration_length;
        step_1 = step_1 > 1.0 ? integration_length : step_1 * integration_length;
        /* Sample at one third of the integrated interval -> better results for exponential functions */
        f32 integration_step = step_0 + (step_1 - step_0) * 0.3;
        f32 d_int_step = step_1 - step_0;

        /* Position shift */
        f32vec3 new_position = world_position + integration_step * world_direction;
        ScatteringSample medium_scattering = sample_medium_scattering_detailed(gpu_input, new_position);
        f32vec3 medium_extinction = sample_medium_extinction(gpu_input, new_position);

        f32vec3 up_vector = normalize(new_position);
        TransmittanceParams transmittance_lut_params = TransmittanceParams(length(new_position), dot(sun_direction, up_vector));

        /* uv coordinates later used to sample transmittance texture */
        f32vec2 trans_texture_uv = transmittance_lut_to_uv(transmittance_lut_params, deref(gpu_input).sky_settings.atmosphere_bottom, deref(gpu_input).sky_settings.atmosphere_top);
        f32vec3 transmittance_to_sun = texture(daxa_sampler2D(transmittance_lut, deref(gpu_input).sampler_llc), trans_texture_uv).rgb;

        f32vec3 phase_times_scattering = medium_scattering.mie * mie_phase_value + medium_scattering.ray * rayleigh_phase_value;

        f32 earth_intersection_distance = ray_sphere_intersect_nearest(
            new_position, sun_direction, planet_zero, deref(gpu_input).sky_settings.atmosphere_bottom);
        f32 in_earth_shadow = earth_intersection_distance == -1.0 ? 1.0 : 0.0;

        f32vec3 multiscattered_luminance = get_multiple_scattering(new_position, dot(sun_direction, up_vector)); 

        /* Light arriving from the sun to this point */
        f32vec3 sun_light = in_earth_shadow * transmittance_to_sun * phase_times_scattering +
            multiscattered_luminance * (medium_scattering.ray + medium_scattering.mie);

        /* TODO: This probably should be a texture lookup*/
        f32vec3 trans_increase_over_integration_step = exp(-(medium_extinction * d_int_step));

        f32vec3 sun_light_integ = (sun_light - sun_light * trans_increase_over_integration_step) / medium_extinction;

        if(medium_extinction.r == 0.0) { sun_light_integ.r = 0.0; }
        if(medium_extinction.g == 0.0) { sun_light_integ.g = 0.0; }
        if(medium_extinction.b == 0.0) { sun_light_integ.b = 0.0; }

        accum_light += accum_transmittance * sun_light_integ;
        accum_transmittance *= trans_increase_over_integration_step;
    }
    return accum_light;
}

void main()
{
    if (any(greaterThan(gl_GlobalInvocationID.xy, SKY_SKY_RES))) {
        return;
    }

    // Hardcode player position to be 100 meters above sea level
    f32vec3 world_position = f32vec3(0.0, 0.0, deref(gpu_input).sky_settings.atmosphere_bottom + 0.1);

    f32vec2 uv = f32vec2(gl_GlobalInvocationID.xy) / f32vec2(SKY_SKY_RES);
    SkyviewParams skyview_params = uv_to_skyview_lut_params(
        uv,
        deref(gpu_input).sky_settings.atmosphere_bottom,
        deref(gpu_input).sky_settings.atmosphere_top,
        SKY_SKY_RES,
        length(world_position)
    );

    f32 sun_zenith_cos_angle = dot(normalize(world_position), deref(gpu_input).sky_settings.sun_direction);
    // sin^2 + cos^2 = 1 -> sqrt(1 - cos^2) = sin
    // rotate the sun direction so that we are aligned with the y = 0 axis
    f32vec3 local_sun_direction = normalize(f32vec3(
        safe_sqrt(1.0 - sun_zenith_cos_angle * sun_zenith_cos_angle),
        0.0,
        sun_zenith_cos_angle));
    
    f32vec3 world_direction = f32vec3(
        cos(skyview_params.light_view_angle) * sin(skyview_params.view_zenith_angle),
        sin(skyview_params.light_view_angle) * sin(skyview_params.view_zenith_angle),
        cos(skyview_params.view_zenith_angle));

    if (!move_to_top_atmosphere(world_position, world_direction, deref(gpu_input).sky_settings.atmosphere_bottom, deref(gpu_input).sky_settings.atmosphere_top))
    {
        /* No intersection with the atmosphere */
        imageStore(daxa_image2D(sky_lut), i32vec2(gl_GlobalInvocationID.xy), f32vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }
    f32vec3 luminance = integrate_scattered_luminance(world_position, world_direction, local_sun_direction, 30);
    imageStore(daxa_image2D(sky_lut), i32vec2(gl_GlobalInvocationID.xy), f32vec4(luminance, 1.0));
}

#endif

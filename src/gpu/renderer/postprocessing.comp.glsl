#include <shared/app.inl>
#include <utils/math.glsl>

#if COMPOSITING_COMPUTE
#include <utils/sky.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    daxa_f32vec4 output_tex_size = daxa_f32vec4(deref(gpu_input).frame_dim, 0, 0);
    output_tex_size.zw = daxa_f32vec2(1.0, 1.0) / output_tex_size.xy;
    daxa_f32vec2 uv = get_uv(gl_GlobalInvocationID.xy, output_tex_size);

    ViewRayContext vrc = vrc_from_uv(globals, uv_to_ss(gpu_input, uv, output_tex_size));
    daxa_f32vec3 ray_dir = ray_dir_ws(vrc);

    daxa_u32vec4 g_buffer_nrm_samples = textureGather(
        daxa_usampler2D(g_buffer_image_id, deref(gpu_input).sampler_llc),
        daxa_f32vec2(gl_GlobalInvocationID.xy) / deref(gpu_input).frame_dim.xy, 1);

    daxa_u32vec4 g_buffer_value = texelFetch(daxa_utexture2D(g_buffer_image_id), daxa_i32vec2(gl_GlobalInvocationID.xy), 0);
    // daxa_f32vec3 nrm = u16_to_nrm(g_buffer_value.y);

    daxa_f32vec3 nrm = u16_to_nrm(g_buffer_nrm_samples.x) + u16_to_nrm(g_buffer_nrm_samples.y) + u16_to_nrm(g_buffer_nrm_samples.z) + u16_to_nrm(g_buffer_nrm_samples.w);
    nrm = normalize(nrm);

    daxa_f32 depth = uintBitsToFloat(g_buffer_value.z);

    daxa_i32vec2 shadow_coord = daxa_i32vec2(gl_GlobalInvocationID.xy);
    uint shadow_index = shadow_coord.x + shadow_coord.y * uint(deref(gpu_input).frame_dim.x);
    uint shadow_value = (deref(shadow_image_buffer[shadow_index / 32]) >> (shadow_index & 31)) & 1;

    daxa_f32vec4 shaded_value = daxa_f32vec4(shadow_value); // texelFetch(daxa_texture2D(shading_image_id), , 0);
    shaded_value *= clamp(dot(nrm, SUN_DIR), 0.0, 1.0);

    if (depth != 0) {
        ray_dir = SUN_DIR;
    }
    AtmosphereLightingInfo sky_lighting = get_atmosphere_lighting(sky_lut, transmittance_lut, ray_dir, nrm);

    daxa_f32vec3 ssao_value;
    {
        ssao_value = texelFetch(daxa_texture2D(ssao_image_id), daxa_i32vec2(gl_GlobalInvocationID.xy), 0).rrr;
        ssao_value = pow(ssao_value, vec3(2)) * 2.0;
        ssao_value *= sky_lighting.atmosphere_normal_illuminance * (dot(nrm, vec3(0, 0, 1)) * 0.5 + 0.5);
    }
    // daxa_f32vec4 temp_val = texelFetch(daxa_texture2D(indirect_diffuse_image_id), daxa_i32vec2(gl_GlobalInvocationID.xy), 0);
    daxa_f32vec4 particles_color = vec4(0); // texelFetch(daxa_texture2D(particles_image_id), daxa_i32vec2(gl_GlobalInvocationID.xy), 0);
    daxa_f32vec3 direct_value = shaded_value.xyz * (sky_lighting.atmosphere_direct_illuminance + sky_lighting.sun_direct_illuminance);

    daxa_f32vec3 emit_col = uint_urgb9e5_to_f32vec3(g_buffer_value.w);
    if (depth == 0) {
        emit_col += sky_lighting.atmosphere_direct_illuminance + sky_lighting.sun_direct_illuminance;
    }

    daxa_f32vec3 albedo_col = (uint_rgba8_to_f32vec4(g_buffer_value.x).rgb);

    daxa_f32vec3 lighting = daxa_f32vec3(0.0);
    // Direct sun illumination
    lighting += direct_value;
    // Sky ambient
    lighting += daxa_f32vec3(ssao_value);
    // Default ambient
    // lighting += 1.0;

    daxa_f32vec3 final_color = particles_color.rgb + emit_col + albedo_col * lighting;
    final_color *= 0.05;

    imageStore(daxa_image2D(dst_image_id), daxa_i32vec2(gl_GlobalInvocationID.xy), daxa_f32vec4(final_color, 1.0));
}
#endif

#if POSTPROCESSING_RASTER

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
    x = x * 0.2;
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

    // daxa_u32vec4 g_buffer_value = texelFetch(daxa_utexture2D(g_buffer_image_id), daxa_i32vec2(gl_FragCoord.xy), 0);
    // daxa_u32vec4 g_buffer_value = texture(daxa_usampler2D(g_buffer_image_id, deref(gpu_input).sampler_llc), (uv + 0.5) / deref(gpu_input).frame_dim.xy);
    // daxa_f32vec3 nrm = u16_to_nrm(g_buffer_value.y);
    // daxa_f32 depth = uintBitsToFloat(g_buffer_value.z);
    // final_color = nrm;
    // final_color = max(dot(normalize(nrm), deref(gpu_input).sky_settings.sun_direction), vec3(0.0));
    // final_color = vec3(dot(normalize(nrm), deref(gpu_input).sky_settings.sun_direction.xyz));

    color = daxa_f32vec4(color_correct(final_color), 1.0);
}

#endif

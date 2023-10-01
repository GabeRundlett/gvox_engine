#include <shared/app.inl>
#include <utils/math.glsl>

#if COMPOSITING_COMPUTE
#include <utils/sky.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec4 g_buffer_value = texelFetch(daxa_utexture2D(g_buffer_image_id), i32vec2(gl_GlobalInvocationID.xy), 0);
    f32vec3 nrm = u16_to_nrm(g_buffer_value.y);

    f32vec4 shaded_value = texelFetch(daxa_texture2D(shading_image_id), i32vec2(gl_GlobalInvocationID.xy), 0);
    shaded_value *= clamp(dot(nrm, SUN_DIR), 0.0, 1.0);

    f32vec3 ssao_value;
    if (ENABLE_DIFFUSE_GI) {
        ssao_value = texelFetch(daxa_texture2D(ssao_image_id), i32vec2(gl_GlobalInvocationID.xy / 2), 0).rgb;
    } else {
        ssao_value = texelFetch(daxa_texture2D(ssao_image_id), i32vec2(gl_GlobalInvocationID.xy), 0).rrr * get_far_sky_color(sky_lut, nrm);
    }
    // f32vec4 temp_val = texelFetch(daxa_texture2D(indirect_diffuse_image_id), i32vec2(gl_GlobalInvocationID.xy), 0);
    // f32vec4 particles_color = texelFetch(daxa_texture2D(particles_image_id), i32vec2(gl_GlobalInvocationID.xy), 0);
    f32vec3 direct_value = shaded_value.xyz * SUN_COL(sky_lut);

    f32vec3 emit_col = uint_urgb9e5_to_f32vec3(g_buffer_value.w);

    f32vec3 albedo_col = (uint_rgba8_to_f32vec4(g_buffer_value.x).rgb);

    f32vec3 lighting = f32vec3(0.0);
    // Direct sun illumination
    lighting += direct_value;
    // Sky ambient
    lighting += f32vec3(ssao_value);
    // Default ambient
    // lighting += 2.0;

    f32vec3 final_color = emit_col + albedo_col * lighting;

    imageStore(daxa_image2D(dst_image_id), i32vec2(gl_GlobalInvocationID.xy), f32vec4(final_color, 1.0));
}
#endif

#if POSTPROCESSING_RASTER

DAXA_DECL_PUSH_CONSTANT(PostprocessingRasterPush, push)

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

f32 luminance(f32vec3 color) {
    f32vec3 luminanceCoefficients = SRGB_2_XYZ_MAT[1];
    return dot(color, luminanceCoefficients);
}

const f32mat3x3 agxTransform = f32mat3x3(
    0.842479062253094, 0.0423282422610123, 0.0423756549057051,
    0.0784335999999992, 0.878468636469772, 0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104);

const f32mat3x3 agxTransformInverse = f32mat3x3(
    1.19687900512017, -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433, 1.15107367264116);

f32vec3 agxDefaultContrastApproximation(f32vec3 x) {
    f32vec3 x2 = x * x;
    f32vec3 x4 = x2 * x2;

    return +15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

void agx(inout f32vec3 color) {
    const f32 minEv = -12.47393;
    const f32 maxEv = 4.026069;

    color = agxTransform * color;
    color = clamp(log2(color), minEv, maxEv);
    color = (color - minEv) / (maxEv - minEv);
    color = agxDefaultContrastApproximation(color);
}

void agxEotf(inout f32vec3 color) {
    color = agxTransformInverse * color;
}

void agxLook(inout f32vec3 color) {
    // Punchy
    const f32vec3 slope = f32vec3(1.1);
    const f32vec3 power = f32vec3(1.2);
    const f32 saturation = 1.3;

    f32 luma = luminance(color);

    // color = pow(color * slope, power);
    color = luma + saturation * (color - luma);
}

const f32 exposureBias = 1.0;
const f32 calibration = 12.5;        // Light meter calibration
const f32 sensorSensitivity = 100.0; // Sensor sensitivity

f32 computeEV100fromLuminance(f32 luminance) {
    return log2(luminance * sensorSensitivity * exposureBias / calibration);
}

f32 computeExposureFromEV100(f32 ev100) {
    return 1.0 / (1.2 * exp2(ev100));
}

f32 computeExposure(f32 averageLuminance) {
    f32 ev100 = computeEV100fromLuminance(averageLuminance);
    f32 exposure = computeExposureFromEV100(ev100);

    return exposure;
}

f32vec3 color_correct(f32vec3 x) {
    // x = x * 0.01;
    // agx(x);
    // agxLook(x);
    // agxEotf(x);
    x = srgb_encode(x);
    return x;
}

layout(location = 0) out f32vec4 color;

void main() {
    f32vec2 g_buffer_scl = f32vec2(deref(gpu_input).render_res_scl) * f32vec2(deref(gpu_input).frame_dim) / f32vec2(deref(gpu_input).rounded_frame_dim);
    f32vec2 uv = f32vec2(gl_FragCoord.xy);
    f32vec3 final_color = texelFetch(daxa_texture2D(composited_image_id), i32vec2(uv), 0).rgb;

    if ((deref(gpu_input).flags & GAME_FLAG_BITS_PAUSED) == 0) {
        i32vec2 center_offset_uv = i32vec2(uv.xy) - i32vec2(deref(gpu_input).frame_dim.xy / deref(gpu_input).render_res_scl) / 2;
        if ((abs(center_offset_uv.x) <= 1 || abs(center_offset_uv.y) <= 1) && abs(center_offset_uv.x) + abs(center_offset_uv.y) < 6) {
            final_color *= f32vec3(0.1);
        }
        if ((abs(center_offset_uv.x) <= 0 || abs(center_offset_uv.y) <= 0) && abs(center_offset_uv.x) + abs(center_offset_uv.y) < 5) {
            final_color += f32vec3(2.0);
        }
    }

    color = f32vec4(color_correct(final_color), 1.0);
}

#endif

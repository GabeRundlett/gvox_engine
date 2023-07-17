#include <shared/shared.inl>
#include <utils/math.glsl>
#include <utils/sky.glsl>

#define FILMIC

DAXA_DECL_PUSH_CONSTANT(PostprocessingRasterPush, push)

f32vec3 srgb_encode(f32vec3 x) {
    return mix(12.92 * x, 1.055 * pow(x, f32vec3(.41666)) - .055, step(.0031308, x));
}

f32vec3 aces_filmic(f32vec3 x) {
    f32 a = 2.51;
    f32 b = 0.03;
    f32 c = 2.43;
    f32 d = 0.59;
    f32 e = 0.14;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

f32vec3 color_correct(f32vec3 x) {
#if defined(FILMIC)
    x = max(x, f32vec3(0, 0, 0));
    x = x * 0.25;
    x = aces_filmic(x);
    x = srgb_encode(x);
#endif
    return x;
}

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_COMPUTE
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 pixel_i = gl_GlobalInvocationID.xy;
    if (pixel_i.x >= deref(gpu_input).frame_dim.x ||
        pixel_i.y >= deref(gpu_input).frame_dim.y)
        return;

    // broken
    f32vec4 final_color = imageLoad(daxa_image2D(g_buffer_image_id), i32vec2(pixel_i));

    imageStore(daxa_image2D(final_image_id), i32vec2(pixel_i), f32vec4(color_correct(final_color.rgb), 0));
}
#endif

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

void main() {
    switch (gl_VertexIndex) {
    case 0: gl_Position = vec4(-1, -1, 0, 1); break;
    case 1: gl_Position = vec4(-1, +4, 0, 1); break;
    case 2: gl_Position = vec4(+4, -1, 0, 1); break;
    }
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT

layout(location = 0) out f32vec4 color;

#define USE_SAMPLER 0

void main() {
    f32vec2 g_buffer_scl = f32vec2(deref(gpu_input).render_res_scl);
    f32vec2 uv = f32vec2(gl_FragCoord.xy);
    u32vec4 g_buffer_value = texelFetch(daxa_utexture2D(g_buffer_image_id), i32vec2(uv * g_buffer_scl), 0);

#if USE_SAMPLER
    i32vec2 g_buffer_size = i32vec2(deref(gpu_input).rounded_frame_dim);
    f32vec2 scl = 1.0 / f32vec2(g_buffer_size);
    f32vec4 shaded_value = texture(daxa_sampler2D(reconstructed_shading_image_id, push.final_sampler), uv * scl);
#else
    f32vec4 shaded_value = texelFetch(daxa_texture2D(reconstructed_shading_image_id), i32vec2(uv * g_buffer_scl), 0);
#endif
    f32 ssao_value = shaded_value.w;
    f32vec3 direct_value = shaded_value.xyz;

    f32vec3 nrm = u16_to_nrm(g_buffer_value.y);
    f32vec3 emit_col = uint_urgb9e5_to_float3(g_buffer_value.w);

    f32vec3 albedo_col = uint_rgba8_to_float4(g_buffer_value.x).rgb;
    f32vec3 final_color = emit_col + albedo_col * (direct_value * max(0.0, dot(nrm, SUN_DIR)) + f32vec3(ssao_value) * sample_sky_ambient(nrm));

    color = f32vec4(color_correct(final_color), 1.0);
}

#endif

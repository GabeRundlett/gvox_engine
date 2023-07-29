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

layout(location = 0) out f32vec4 color;

void main() {
    f32vec2 g_buffer_scl = f32vec2(deref(gpu_input).render_res_scl) * f32vec2(deref(gpu_input).frame_dim) / f32vec2(deref(gpu_input).rounded_frame_dim);
    f32vec2 uv = f32vec2(gl_FragCoord.xy);
    u32vec4 g_buffer_value = texelFetch(daxa_utexture2D(g_buffer_image_id), i32vec2(uv * g_buffer_scl), 0);

    f32 ssao_value = texelFetch(daxa_texture2D(ssao_image_id), i32vec2(uv * g_buffer_scl), 0).x;
    f32vec4 shaded_value = texelFetch(daxa_texture2D(reconstructed_shading_image_id), i32vec2(uv * g_buffer_scl), 0);
    f32vec4 particles_color = texelFetch(daxa_texture2D(particles_image_id), i32vec2(uv * g_buffer_scl), 0);
    f32vec3 direct_value = shaded_value.xyz;

    f32vec3 nrm = u16_to_nrm(g_buffer_value.y);
    f32vec3 emit_col = uint_urgb9e5_to_f32vec3(g_buffer_value.w);

    f32vec3 albedo_col = (uint_rgba8_to_f32vec4(g_buffer_value.x).rgb);
    f32vec3 final_color = particles_color.rgb + emit_col + albedo_col * (direct_value * max(0.0, dot(nrm, SUN_DIR)) + f32vec3(ssao_value) * sample_sky_ambient(nrm));

    color = f32vec4(color_correct(final_color), 1.0);
}

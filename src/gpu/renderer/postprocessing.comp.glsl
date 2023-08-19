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
        ssao_value = texelFetch(daxa_texture2D(ssao_image_id), i32vec2(gl_GlobalInvocationID.xy), 0).rrr * sample_sky_ambient(nrm);
    }
    // f32vec4 temp_val = texelFetch(daxa_texture2D(indirect_diffuse_image_id), i32vec2(gl_GlobalInvocationID.xy), 0);
    // f32vec4 particles_color = texelFetch(daxa_texture2D(particles_image_id), i32vec2(gl_GlobalInvocationID.xy), 0);
    f32vec3 direct_value = shaded_value.xyz;

    f32vec3 emit_col = uint_urgb9e5_to_f32vec3(g_buffer_value.w);

    f32vec3 albedo_col = (uint_rgba8_to_f32vec4(g_buffer_value.x).rgb);

    f32vec3 lighting = f32vec3(0.0);
    // Direct sun illumination
    lighting += direct_value * 1.0;
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

f32vec3 srgb_encode(f32vec3 x) {
    return mix(12.92 * x, 1.055 * pow(x, f32vec3(.41666)) - .055, step(.0031308, x));
}

f32vec3 color_correct(f32vec3 x) {
    x = x * 0.25;
    x = srgb_encode(x);
    return x;
}

layout(location = 0) out f32vec4 color;

void main() {
    f32vec2 g_buffer_scl = f32vec2(deref(gpu_input).render_res_scl) * f32vec2(deref(gpu_input).frame_dim) / f32vec2(deref(gpu_input).rounded_frame_dim);
    f32vec2 uv = f32vec2(gl_FragCoord.xy);
    f32vec3 final_color = texelFetch(daxa_texture2D(composited_image_id), i32vec2(uv), 0).rgb;

    if ((deref(gpu_input).flags & GAME_FLAG_BITS_PAUSED) == 0) {
        i32vec2 center_offset_uv = i32vec2(uv.xy) - i32vec2(deref(gpu_input).frame_dim.xy) / 2;
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

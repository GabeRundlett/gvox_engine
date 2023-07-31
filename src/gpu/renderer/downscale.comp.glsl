#include <shared/shared.inl>
#include <utils/downscale.glsl>
#include <utils/math.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 offset = get_downscale_offset(gpu_input);
    i32vec2 src_px = i32vec2(gl_GlobalInvocationID.xy * SHADING_SCL + offset);

#if DOWNSCALE_DEPTH
    f32vec4 output_val = texelFetch(daxa_texture2D(src_image_id), src_px, 0);
#elif DOWNSCALE_NRM
    u32vec4 g_buffer_value = texelFetch(daxa_utexture2D(src_image_id), src_px, 0);
    f32vec3 normal_ws = u16_to_nrm_unnormalized(g_buffer_value.y);
    f32vec3 normal_vs = normalize((deref(globals).player.cam.world_to_view * f32vec4(normal_ws, 0)).xyz);
    f32vec4 output_val = f32vec4(normal_vs, 1);
#endif

    imageStore(daxa_image2D(dst_image_id), i32vec2(gl_GlobalInvocationID.xy), output_val);
}

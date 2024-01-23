#include <shared/app.inl>
#include <utils/downscale.glsl>
#include <utils/math.glsl>
#include <utils/safety.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    daxa_u32vec2 offset = get_downscale_offset(gpu_input);
    daxa_i32vec2 src_px = daxa_i32vec2(gl_GlobalInvocationID.xy * SHADING_SCL + offset);

#if DOWNSCALE_DEPTH || DOWNSCALE_SSAO
    daxa_f32vec4 output_val = safeTexelFetch(src_image_id, src_px, 0);
#elif DOWNSCALE_NRM
    daxa_u32vec4 g_buffer_value = safeTexelFetchU(src_image_id, src_px, 0);
    daxa_f32vec3 normal_ws = u16_to_nrm_unnormalized(g_buffer_value.y);
    daxa_f32vec3 normal_vs = normalize((deref(globals).player.cam.world_to_view * daxa_f32vec4(normal_ws, 0)).xyz);
    daxa_f32vec4 output_val = daxa_f32vec4(normal_vs, 1);
#endif

    safeImageStore(dst_image_id, daxa_i32vec2(gl_GlobalInvocationID.xy), output_val);
}

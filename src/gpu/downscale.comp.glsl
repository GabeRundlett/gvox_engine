#include <shared/shared.inl>
#include <utils/downscale.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 offset = get_downscale_offset(gpu_input);
    f32vec4 src_val = texelFetch(daxa_texture2D(src_image_id), i32vec2(gl_GlobalInvocationID.xy * SHADING_SCL + offset), 0);

    imageStore(daxa_image2D(dst_image_id), i32vec2(gl_GlobalInvocationID.xy), src_val);
}

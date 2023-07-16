#include <shared/shared.inl>
#include <utils/downscale.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 offset = get_downscale_offset(gpu_input);

    f32 ssao_value = imageLoad(daxa_image2D(ssao_image_id), i32vec2(gl_GlobalInvocationID.xy)).r;
    f32vec3 direct_value = imageLoad(daxa_image2D(indirect_diffuse_image_id), i32vec2(gl_GlobalInvocationID.xy)).rgb;

    imageStore(daxa_image2D(dst_image_id), i32vec2(gl_GlobalInvocationID.xy * SHADING_SCL + offset), f32vec4(direct_value, ssao_value));
}

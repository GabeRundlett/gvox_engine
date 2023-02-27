#include <shared/shared.inl>

#include <utils/trace.glsl>

DAXA_USE_PUSH_CONSTANT(TracePrimaryComputePush)

#define INPUT deref(daxa_push_constant.gpu_input)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 pixel_i = gl_GlobalInvocationID.xy;
    if (pixel_i.x >= INPUT.frame_dim.x ||
        pixel_i.y >= INPUT.frame_dim.y)
        return;

    f32vec2 pixel_p = pixel_i;
    f32vec2 frame_dim = INPUT.frame_dim;
    f32vec2 inv_frame_dim = f32vec2(1.0, 1.0) / frame_dim;
    f32 aspect = frame_dim.x * inv_frame_dim.y;

    f32vec2 uv = pixel_p * inv_frame_dim;

    uv = (uv - 0.5) * f32vec2(aspect, 1.0) * 2.0;
    f32vec3 ray_pos = create_view_pos(daxa_push_constant.gpu_globals);
    f32vec3 ray_dir = create_view_dir(daxa_push_constant.gpu_globals, uv);
    trace(daxa_push_constant.gpu_gvox_model, ray_pos, ray_dir);

    imageStore(
        daxa_push_constant.render_pos_image_id,
        i32vec2(pixel_i),
        f32vec4(ray_pos, 0));
}
#undef INPUT

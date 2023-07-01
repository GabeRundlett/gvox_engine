#include <shared/shared.inl>

#include <utils/trace.glsl>

#define USE_BLUE_NOISE 1

#define SETTINGS deref(settings)
#define INPUT deref(gpu_input)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 pixel_i = gl_GlobalInvocationID.xy;

    f32vec2 pixel_p = f32vec2(pixel_i) + 0.5;
    f32vec2 frame_dim = f32vec2(INPUT.frame_dim) / PREPASS_SCL;
    f32vec2 inv_frame_dim = f32vec2(1.0, 1.0) / frame_dim;
    f32 aspect = frame_dim.x * inv_frame_dim.y;

    f32vec2 uv = pixel_p * inv_frame_dim;

    uv = (uv - 0.5) * f32vec2(aspect, 1.0) * 2.0;
    f32vec3 cam_pos = create_view_pos(deref(globals).player);
    f32vec3 ray_pos = cam_pos;
    f32vec3 ray_dir = create_view_dir(deref(globals).player, uv);
    u32vec3 chunk_n = u32vec3(1u << SETTINGS.log2_chunks_per_axis);

    u32 step_n = trace(voxel_malloc_global_allocator, voxel_chunks, chunk_n, ray_pos, ray_dir);

    f32 depth = length(ray_pos - cam_pos);

    imageStore(daxa_image2D(render_depth_prepass_image), i32vec2(pixel_i), f32vec4(depth, step_n, 0, 0));
}
#undef INPUT
#undef SETTINGS

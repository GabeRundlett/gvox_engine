#include <shared/shared.inl>

#include <utils/trace.glsl>

#define SETTINGS deref(settings)
#define INPUT deref(gpu_input)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 pixel_i = gl_GlobalInvocationID.xy;

    f32vec2 pixel_p = f32vec2(pixel_i) + 0.5;
    f32vec2 frame_dim = INPUT.frame_dim;
    f32vec2 inv_frame_dim = f32vec2(1.0, 1.0) / frame_dim;
    f32 aspect = frame_dim.x * inv_frame_dim.y;

#if RANDOMIZE_PRIMARY_RAY_DIR
#if USE_BLUE_NOISE
    f32vec2 blue_noise = texelFetch(daxa_texture3D(blue_noise_vec2), ivec3(pixel_i, INPUT.frame_index) & ivec3(127, 127, 63), 0).xy - 0.5;
    pixel_p += blue_noise * 1.0;
#else
    rand_seed(pixel_i.x + pixel_i.y * INPUT.frame_dim.x + u32(INPUT.time * 719393));
    f32vec2 uv_offset = f32vec2(rand(), rand()) - 0.5;
    pixel_p += uv_offset * 1.0;
#endif
#endif

    f32vec2 uv = pixel_p * inv_frame_dim;

    uv = (uv - 0.5) * f32vec2(aspect, 1.0) * 2.0;

    f32vec3 ray_dir = create_view_dir(deref(globals).player, uv);

#if !ENABLE_DEPTH_PREPASS
    f32 prepass_depth = 0.0;
    f32 prepass_steps = 0.0;
#else
    f32 max_depth = MAX_SD;
    f32 prepass_depth = max_depth;
    f32 prepass_steps = 0.0;

    for (i32 yi = -1; yi <= 1; ++yi) {
        for (i32 xi = -1; xi <= 1; ++xi) {
            i32vec2 pt = i32vec2(pixel_i / PREPASS_SCL) + i32vec2(xi, yi);
            pt = clamp(pt, i32vec2(0), i32vec2(INPUT.rounded_frame_dim / PREPASS_SCL));
            f32vec2 prepass_data = imageLoad(daxa_image2D(render_depth_prepass_image), pt).xy;
            f32 loaded_depth = prepass_data.x - 1.0 / VOXEL_SCL;
            prepass_depth = max(min(prepass_depth, loaded_depth), 0);
            if (prepass_depth == loaded_depth || prepass_depth == max_depth) {
                prepass_steps = prepass_data.y / 4.0;
            }
        }
    }
#endif

    f32vec3 ray_pos = create_view_pos(deref(globals).player) + ray_dir * prepass_depth;
    u32vec3 chunk_n = u32vec3(1u << SETTINGS.log2_chunks_per_axis);

    u32 step_n = trace(voxel_malloc_global_allocator, voxel_chunks, chunk_n, ray_pos, ray_dir, 0.0);

    imageStore(daxa_image2D(render_pos_image_id), i32vec2(pixel_i), f32vec4(ray_pos, step_n + prepass_steps));
}
#undef INPUT
#undef SETTINGS

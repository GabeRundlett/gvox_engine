#include <shared/shared.inl>

#include <utils/trace.glsl>

#define SKY_COL (f32vec3(0.02, 0.05, 0.90) * 4)
#define SKY_COL_B (f32vec3(0.11, 0.10, 0.54))

// #define SUN_TIME (deref(gpu_input).time)
#define SUN_TIME 0.9
#define SUN_COL (f32vec3(1, 0.90, 0.4) * 20)
#define SUN_DIR normalize(f32vec3(1.2 * abs(sin(SUN_TIME)), -cos(SUN_TIME), abs(sin(SUN_TIME))))

f32vec3 sample_sky_ambient(f32vec3 nrm) {
    f32 sun_val = dot(nrm, SUN_DIR) * 0.1 + 0.06;
    sun_val = pow(sun_val, 2) * 0.2;
    f32 sky_val = clamp(dot(nrm, f32vec3(0, 0, -1)) * 0.2 + 0.5, 0, 1);
    return mix(SKY_COL + sun_val * SUN_COL, SKY_COL_B, pow(sky_val, 2));
}

f32vec3 sample_sky(f32vec3 nrm) {
    f32vec3 light = sample_sky_ambient(nrm);
    f32 sun_val = dot(nrm, SUN_DIR) * 0.5 + 0.5;
    sun_val = sun_val * 200 - 199;
    sun_val = pow(clamp(sun_val * 1.1, 0, 1), 200);
    light += sun_val * SUN_COL;
    return light;
}

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

    VoxelTraceResult trace_result = trace_hierarchy_traversal(VoxelTraceInfo(voxel_malloc_global_allocator, voxel_chunks, chunk_n, ray_dir, MAX_STEPS, MAX_SD, 0.0, true), ray_pos);
    u32 step_n = trace_result.step_n;

    u32vec4 output_value = u32vec4(0);
    // output_value.x = step_n + u32(prepass_steps);

    f32 depth = trace_result.dist + prepass_depth;

    if (depth >= MAX_SD) {
        f32vec3 sky_col = sample_sky(ray_dir);
        output_value.w = float3_to_uint_urgb9e5(sky_col / 1.0);
        output_value.y |= nrm_to_u16(f32vec3(0));
        depth = 0;
    } else {
        output_value.x = trace_result.voxel_data;
        output_value.y |= nrm_to_u16(trace_result.nrm);
    }

    imageStore(daxa_uimage2D(g_buffer_image_id), i32vec2(pixel_i), output_value);
    imageStore(daxa_image2D(depth_image_id), i32vec2(pixel_i), f32vec4(depth, 0, 0, 0));
}
#undef INPUT
#undef SETTINGS

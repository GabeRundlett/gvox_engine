#include <shared/app.inl>

#include <voxels/core.glsl>

#if TRACE_DEPTH_PREPASS_COMPUTE

#define INPUT deref(gpu_input)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    f32vec4 output_tex_size;
    output_tex_size.xy = deref(gpu_input).frame_dim;
    output_tex_size.zw = f32vec2(1.0, 1.0) / output_tex_size.xy;
    f32vec2 uv = get_uv(gl_GlobalInvocationID.xy * SHADING_SCL, output_tex_size);

    ViewRayContext vrc = vrc_from_uv(globals, uv_to_ss(gpu_input, uv, output_tex_size));
    f32vec3 ray_dir = ray_dir_ws(vrc);
    f32vec3 cam_pos = ray_origin_ws(vrc);
    f32vec3 ray_pos = cam_pos;
    u32vec3 chunk_n = u32vec3(1u << deref(gpu_input).log2_chunks_per_axis);

#if ENABLE_DEPTH_PREPASS
    VoxelTraceResult trace_result = voxel_trace(VoxelTraceInfo(VOXEL_TRACE_INFO_PTRS, chunk_n, ray_dir, MAX_STEPS, MAX_DIST, 32.0 * output_tex_size.w * deref(globals).player.cam.clip_to_view[1][1], true), ray_pos);
    u32 step_n = trace_result.step_n;
#else
    u32 step_n = 0;
#endif

    f32 depth = length(ray_pos - cam_pos);

    imageStore(daxa_image2D(render_depth_prepass_image), i32vec2(gl_GlobalInvocationID.xy), f32vec4(depth, step_n, 0, 0));
}
#undef INPUT

#endif

#if TRACE_PRIMARY_COMPUTE

#include <utils/sky.glsl>

#define PIXEL_I gl_GlobalInvocationID.xy
#define INPUT deref(gpu_input)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    f32vec4 output_tex_size = f32vec4(deref(gpu_input).frame_dim, 0, 0);
    output_tex_size.zw = f32vec2(1.0, 1.0) / output_tex_size.xy;
    f32vec2 uv = get_uv(PIXEL_I, output_tex_size);

    ViewRayContext vrc = vrc_from_uv(globals, uv_to_ss(gpu_input, uv, output_tex_size));
    f32vec3 ray_dir = ray_dir_ws(vrc);
    f32vec3 cam_pos = ray_origin_ws(vrc);

#if !ENABLE_DEPTH_PREPASS
    f32 prepass_depth = 0.0;
    f32 prepass_steps = 0.0;
#else
    f32 max_depth = MAX_DIST;
    f32 prepass_depth = max_depth;
    f32 prepass_steps = 0.0;

    for (i32 yi = -1; yi <= 1; ++yi) {
        for (i32 xi = -1; xi <= 1; ++xi) {
            i32vec2 pt = i32vec2(PIXEL_I / PREPASS_SCL) + i32vec2(xi, yi);
            pt = clamp(pt, i32vec2(0), i32vec2(INPUT.rounded_frame_dim / PREPASS_SCL));
            f32vec2 prepass_data = texelFetch(daxa_texture2D(render_depth_prepass_image), pt, 0).xy;
            f32 loaded_depth = prepass_data.x - 1.0 / VOXEL_SCL;
            prepass_depth = max(min(prepass_depth, loaded_depth), 0);
            if (prepass_depth == loaded_depth || prepass_depth == max_depth) {
                prepass_steps = prepass_data.y / 4.0;
            }
        }
    }
#endif

    f32vec3 ray_pos = cam_pos + ray_dir * prepass_depth;
    u32vec3 chunk_n = u32vec3(1u << deref(gpu_input).log2_chunks_per_axis);

    VoxelTraceResult trace_result = voxel_trace(VoxelTraceInfo(VOXEL_TRACE_INFO_PTRS, chunk_n, ray_dir, MAX_STEPS, MAX_DIST, 0.0, true), ray_pos);
    u32 step_n = trace_result.step_n;

    f32vec3 chunk_offset_delta = f32vec3(deref(globals).player.chunk_offset - deref(globals).player.prev_chunk_offset) * CHUNK_WORLDSPACE_SIZE;

    u32vec4 output_value = u32vec4(0);

    // f32 depth = length(cam_pos - ray_pos);
    // transform depth to ndc space
    vec4 vs_pos = (deref(globals).player.cam.world_to_view * vec4(ray_pos, 1));
    vec4 prev_vs_pos = (deref(globals).player.cam.world_to_view * vec4(ray_pos + chunk_offset_delta, 1)); // when animated objects exist, this is where they'd put their velocity
    vec4 cs_pos = (deref(globals).player.cam.view_to_sample * vs_pos);
    f32 depth = cs_pos.z / cs_pos.w;
    f32vec3 vs_nrm = f32vec3(0);
    f32vec3 vs_velocity = f32vec3(0);

    if (trace_result.dist == MAX_DIST) {
        f32vec3 sky_col = sample_sky(ray_dir);
        output_value.w = f32vec3_to_uint_urgb9e5(sky_col / 1.0);
        output_value.y |= nrm_to_u16(f32vec3(0));
        depth = 0.0;
    } else {
        output_value.x = trace_result.voxel_data;
        output_value.y |= nrm_to_u16(trace_result.nrm);
        vs_nrm = (deref(globals).player.cam.world_to_view * f32vec4(trace_result.nrm, 0)).xyz;
        vs_velocity = (prev_vs_pos.xyz / prev_vs_pos.w) - (vs_pos.xyz / vs_pos.w);
    }

    imageStore(daxa_uimage2D(g_buffer_image_id), i32vec2(gl_GlobalInvocationID.xy), output_value);
    imageStore(daxa_image2D(vs_normal_image_id), i32vec2(gl_GlobalInvocationID.xy), f32vec4(vs_nrm * 0.5 + 0.5, 0));
    imageStore(daxa_image2D(velocity_image_id), i32vec2(gl_GlobalInvocationID.xy), f32vec4(vs_velocity, 0));
    imageStore(daxa_image2D(depth_image_id), i32vec2(gl_GlobalInvocationID.xy), f32vec4(depth, 0, 0, 0));
}
#undef INPUT

#endif

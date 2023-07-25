#include <shared/shared.inl>

#include <utils/trace.glsl>

#define SETTINGS deref(settings)
#define INPUT deref(gpu_input)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    f32vec4 output_tex_size;
    output_tex_size.xy = deref(gpu_input).frame_dim;
    output_tex_size.zw = f32vec2(1.0, 1.0) / output_tex_size.xy;
    f32vec2 uv = get_uv(gl_GlobalInvocationID.xy * SHADING_SCL, output_tex_size);

    ViewRayContext vrc = vrc_from_uv(globals, uv);
    f32vec3 ray_dir = ray_dir_ws(vrc);
    f32vec3 cam_pos = ray_origin_ws(vrc);
    f32vec3 ray_pos = cam_pos;
    u32vec3 chunk_n = u32vec3(1u << SETTINGS.log2_chunks_per_axis);

#if ENABLE_DEPTH_PREPASS
    VoxelTraceResult trace_result = trace_hierarchy_traversal(VoxelTraceInfo(voxel_malloc_page_allocator, voxel_chunks, chunk_n, ray_dir, MAX_STEPS, MAX_DIST, 32.0 * output_tex_size.w * deref(globals).player.cam.clip_to_view[1][1], true), ray_pos);
    u32 step_n = trace_result.step_n;
#else
    u32 step_n = 0;
#endif

    f32 depth = length(ray_pos - cam_pos);

    imageStore(daxa_image2D(render_depth_prepass_image), i32vec2(gl_GlobalInvocationID.xy), f32vec4(depth, step_n, 0, 0));
}
#undef INPUT
#undef SETTINGS

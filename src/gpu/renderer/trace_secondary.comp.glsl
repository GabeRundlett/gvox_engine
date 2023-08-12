#include <shared/app.inl>

#include <utils/trace.glsl>
#include <utils/sky.glsl>
#include <utils/downscale.glsl>

#if TRACE_SECONDARY_COMPUTE

#define INPUT deref(gpu_input)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    f32vec4 output_tex_size;
    output_tex_size.xy = deref(gpu_input).frame_dim;
    output_tex_size.zw = f32vec2(1.0, 1.0) / output_tex_size.xy;
    u32vec2 offset = get_downscale_offset(gpu_input);
    f32vec2 uv = get_uv(gl_GlobalInvocationID.xy * SHADING_SCL + offset, output_tex_size);

    f32 depth = texelFetch(daxa_texture2D(depth_image_id), i32vec2(gl_GlobalInvocationID.xy), 0).r;
    u32vec4 g_buffer_value = texelFetch(daxa_utexture2D(g_buffer_image_id), i32vec2(gl_GlobalInvocationID.xy * SHADING_SCL + offset), 0);
    f32vec3 nrm = u16_to_nrm(g_buffer_value.y);

    ViewRayContext vrc = vrc_from_uv_and_depth(globals, uv_to_ss(gpu_input, uv, output_tex_size), depth);
    f32vec3 cam_dir = ray_dir_ws(vrc);
    f32vec3 cam_pos = ray_origin_ws(vrc);
    f32vec3 ray_pos = ray_hit_ws(vrc);

    f32vec2 blue_noise = texelFetch(daxa_texture3D(blue_noise_vec2), ivec3(gl_GlobalInvocationID.xy, INPUT.frame_index) & ivec3(127, 127, 63), 0).xy - 0.5;

    if (depth == 0.0 || dot(nrm, nrm) == 0.0) {
        imageStore(daxa_image2D(indirect_diffuse_image_id), i32vec2(gl_GlobalInvocationID.xy), f32vec4(0, 0, 0, 0));
        return;
    }

    u32vec3 chunk_n = u32vec3(1u << deref(gpu_input).log2_chunks_per_axis);

    mat3 tbn = tbn_from_normal(SUN_DIR);
    f32vec3 ray_dir = tbn * normalize(vec3((rand_circle_pt(abs(blue_noise)) - 0.5) * tan(SUN_ANGULAR_DIAMETER), 1));
    // f32vec3 ray_dir = SUN_DIR;

    VoxelTraceResult trace_result = trace_hierarchy_traversal(VoxelTraceInfo(VOXEL_TRACE_INFO_PTRS, chunk_n, ray_dir, MAX_STEPS, MAX_DIST, 0.0, true), ray_pos);

    f32vec3 col = SUN_COL * f32(trace_result.dist == MAX_DIST);

    imageStore(daxa_image2D(indirect_diffuse_image_id), i32vec2(gl_GlobalInvocationID.xy), f32vec4(col, 0));
}
#undef INPUT

#endif

#if UPSCALE_RECONSTRUCT_COMPUTE

f32vec4 get_prev_val(i32vec2 pixel_i) {
    return texelFetch(daxa_texture2D(src_image_id), pixel_i, 0);
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 offset = get_downscale_offset(gpu_input);
    u32vec2 rounded_frame_dim = deref(gpu_input).rounded_frame_dim;
    u32vec2 half_frame_dim = rounded_frame_dim / 2;
    u32vec2 tile = u32vec2(gl_GlobalInvocationID.x >= half_frame_dim.x, gl_GlobalInvocationID.y >= half_frame_dim.y);
    u32vec2 in_tile_i = gl_GlobalInvocationID.xy - tile * half_frame_dim;
    i32vec2 output_i = i32vec2(in_tile_i * SHADING_SCL + tile);

    f32vec4 result = f32vec4(0.0);

    if (deref(gpu_input).resize_factor != 0.0) {
        f32vec4 reproj_val = texelFetch(daxa_texture2D(reprojection_image_id), output_i, 0);
        f32vec4 prev_val = get_prev_val(output_i + i32vec2(reproj_val.xy * 0.0 * deref(gpu_input).frame_dim));
        f32 validity = reproj_val.z;
        if (offset == tile) {
            f32 ssao_value = 0.0; // texelFetch(daxa_texture2D(ssao_image_id), i32vec2(in_tile_i), 0).r;
            f32vec3 direct_value = texelFetch(daxa_texture2D(scaled_shading_image), i32vec2(in_tile_i), 0).rgb;
            result = f32vec4(direct_value, ssao_value);
            // result = mix(prev_val, f32vec4(direct_value, ssao_value), 0.05);
        } else {
            result = prev_val;
        }
    }

    imageStore(daxa_image2D(dst_image_id), output_i, result);
}

#endif

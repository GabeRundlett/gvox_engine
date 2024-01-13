#include <shared/app.inl>

#include <utils/math.glsl>
#include <voxels/core.glsl>
#include <utils/sky.glsl>
#include <utils/downscale.glsl>

#if TRACE_SECONDARY_COMPUTE

#define INPUT deref(gpu_input)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    daxa_f32vec4 output_tex_size;
    output_tex_size.xy = deref(gpu_input).frame_dim;
    output_tex_size.zw = daxa_f32vec2(1.0, 1.0) / output_tex_size.xy;
    daxa_u32vec2 offset = get_downscale_offset(gpu_input);
    daxa_f32vec2 uv = get_uv(gl_GlobalInvocationID.xy * SHADING_SCL + offset, output_tex_size);

    daxa_f32 depth = texelFetch(daxa_texture2D(depth_image_id), daxa_i32vec2(gl_GlobalInvocationID.xy), 0).r;
    daxa_u32vec4 g_buffer_value = texelFetch(daxa_utexture2D(g_buffer_image_id), daxa_i32vec2(gl_GlobalInvocationID.xy * SHADING_SCL + offset), 0);
    daxa_f32vec3 nrm = u16_to_nrm(g_buffer_value.y);

    ViewRayContext vrc = vrc_from_uv_and_depth(globals, uv_to_ss(gpu_input, uv, output_tex_size), depth);
    daxa_f32vec3 cam_dir = ray_dir_ws(vrc);
    daxa_f32vec3 cam_pos = ray_origin_ws(vrc);
    daxa_f32vec3 ray_pos = ray_hit_ws(vrc) + nrm * 0.001;

    daxa_f32vec2 blue_noise = texelFetch(daxa_texture3D(blue_noise_vec2), ivec3(gl_GlobalInvocationID.xy, INPUT.frame_index) & ivec3(127, 127, 63), 0).xy - 0.5;

    mat3 tbn = tbn_from_normal(SUN_DIR);
    daxa_f32vec3 ray_dir = tbn * normalize(vec3((rand_circle_pt(abs(blue_noise)) - 0.5) * tan(SUN_ANGULAR_DIAMETER), 1));
    // daxa_f32vec3 ray_dir = SUN_DIR;

    ray_pos += ray_dir / depth / (8192.0 * 4.0);

    if (depth == 0.0) { // || dot(ray_dir, nrm) <= 0.0
        return;
    }

    VoxelTraceResult trace_result = voxel_trace(VoxelTraceInfo(VOXELS_BUFFER_PTRS, ray_dir, MAX_STEPS, MAX_DIST, 0.0, true), ray_pos);

    uint out_index = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * uint(output_tex_size.x + SHADING_SCL - 1) / SHADING_SCL;
    uint hit = uint(trace_result.dist == MAX_DIST) << (out_index & 31);
    if (hit != 0) {
        atomicOr(deref(shadow_image_buffer[out_index / 32]), hit);
    }
}
#undef INPUT

#endif

#if UPSCALE_RECONSTRUCT_COMPUTE

daxa_u32 get_prev_val(daxa_i32vec2 pixel_i) {
    if ((pixel_i.x < 0 || pixel_i.y < 0) ||
        (pixel_i.x >= deref(gpu_input).frame_dim.x || pixel_i.y >= deref(gpu_input).frame_dim.y)) {
        return 0;
    }
    daxa_u32 i = pixel_i.x + pixel_i.y * deref(gpu_input).frame_dim.x;
    daxa_u32 index = i / 32;
    daxa_u32 shift = i % 32;
    return (deref(src_image_id[index]) >> shift) & 0x1;
}

daxa_u32 get_scaled_val(daxa_i32vec2 in_tile_i) {
    if ((in_tile_i.x < 0 || in_tile_i.y < 0) ||
        (in_tile_i.x >= deref(gpu_input).frame_dim.x || in_tile_i.y >= deref(gpu_input).frame_dim.y)) {
        return 0;
    }

    daxa_u32 i = in_tile_i.x + in_tile_i.y * (deref(gpu_input).frame_dim.x + SHADING_SCL - 1) / SHADING_SCL;
    daxa_u32 index = i / 32;
    daxa_u32 shift = i % 32;
    return (deref(scaled_shading_image[index]) >> shift) & 0x1;
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    daxa_u32vec2 offset = get_downscale_offset(gpu_input);
    daxa_u32vec2 half_frame_dim = deref(gpu_input).frame_dim / 2;
    daxa_u32vec2 tile = daxa_u32vec2(gl_GlobalInvocationID.x >= half_frame_dim.x, gl_GlobalInvocationID.y >= half_frame_dim.y);
    daxa_u32vec2 in_tile_i = gl_GlobalInvocationID.xy - tile * half_frame_dim;
    daxa_i32vec2 output_i = daxa_i32vec2(in_tile_i * SHADING_SCL + tile);

    daxa_u32 result = 0;          

    if (deref(gpu_input).resize_factor != 0.0) {
        daxa_f32vec4 reproj_val = texelFetch(daxa_texture2D(reprojection_image_id), output_i, 0);
        daxa_u32 prev_val = get_prev_val(output_i + daxa_i32vec2(reproj_val.xy * 0.0 * deref(gpu_input).frame_dim));
        daxa_f32 validity = reproj_val.z;
        if (offset == tile) {
            result = get_scaled_val(daxa_i32vec2(in_tile_i));
            // result = mix(prev_val, result, 0.05);
        } else {
            result = prev_val;
        }
    }

    daxa_u32 i = output_i.x + output_i.y * deref(gpu_input).frame_dim.x;
    daxa_u32 index = i / 32;
    daxa_u32 shift = i % 32;
    result = result << shift;
    if (result != 0) {
        atomicOr(deref(dst_image_id[index]), result);
    }
}

#endif

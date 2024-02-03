#include <shared/app.inl>

#include <utils/math.glsl>
#include <voxels/core.glsl>
#include <utils/sky.glsl>
#include <utils/downscale.glsl>

#if TraceSecondaryComputeShader

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    daxa_f32vec4 output_tex_size = daxa_f32vec4(deref(gpu_input).frame_dim, 0, 0);
    output_tex_size.zw = daxa_f32vec2(1.0, 1.0) / output_tex_size.xy;

    if (any(greaterThanEqual(gl_GlobalInvocationID.xy, uvec2(output_tex_size.xy)))) {
        return;
    }

    daxa_f32vec2 uv = get_uv(gl_GlobalInvocationID.xy, output_tex_size);
    daxa_f32 depth = texelFetch(daxa_texture2D(depth_image_id), daxa_i32vec2(gl_GlobalInvocationID.xy), 0).r;
    daxa_u32vec4 g_buffer_value = texelFetch(daxa_utexture2D(g_buffer_image_id), daxa_i32vec2(gl_GlobalInvocationID.xy), 0);
    daxa_f32vec3 nrm = u16_to_nrm(g_buffer_value.y);

    ViewRayContext vrc = vrc_from_uv_and_depth(globals, uv, depth);
    daxa_f32vec3 cam_dir = ray_dir_ws(vrc);
    daxa_f32vec3 cam_pos = ray_origin_ws(vrc);
    daxa_f32vec3 ray_pos = biased_secondary_ray_origin_ws_with_normal(vrc, nrm);

    daxa_f32vec2 blue_noise = texelFetch(daxa_texture3D(blue_noise_vec2), ivec3(gl_GlobalInvocationID.xy, deref(gpu_input).frame_index) & ivec3(127, 127, 63), 0).yz * 255.0 / 256.0 + 0.5 / 256.0;

    daxa_f32vec3 ray_dir = sample_sun_direction(blue_noise, true);

    uint hit = 0;
    if (depth != 0.0) {
        VoxelTraceResult trace_result = voxel_trace(VoxelTraceInfo(VOXELS_BUFFER_PTRS, ray_dir, MAX_STEPS, MAX_DIST, 0.0, true), ray_pos);
        hit = uint(trace_result.dist == MAX_DIST);
    }

    imageStore(daxa_image2D(shadow_mask), ivec2(gl_GlobalInvocationID.xy), daxa_f32vec4(hit, 0, 0, 0));
}

#endif

#include <shared/app.inl>

#include <utils/math.glsl>
#include <voxels/core.glsl>
#include <utils/sky.glsl>
#include <utils/downscale.glsl>

#if TraceSecondaryComputeShader

vec3 sample_sun_direction(vec2 urand) {
#if !PER_VOXEL_NORMALS
    float sun_angular_radius_cos = deref(gpu_input).sky_settings.sun_angular_radius_cos;
    if (sun_angular_radius_cos < 1.0) {
        const mat3 basis = build_orthonormal_basis(normalize(SUN_DIR));
        return basis * uniform_sample_cone(urand, sun_angular_radius_cos);
    }
#endif
    return SUN_DIR;
}

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

    ViewRayContext vrc = vrc_from_uv_and_depth(globals, uv_to_ss(gpu_input, uv, output_tex_size), depth);
    daxa_f32vec3 cam_dir = ray_dir_ws(vrc);
    daxa_f32vec3 cam_pos = ray_origin_ws(vrc);
#if PER_VOXEL_NORMALS
    daxa_f32vec3 ray_pos = floor(ray_hit_ws(vrc) * VOXEL_SCL) / VOXEL_SCL + 0.5 / VOXEL_SCL + nrm * 1.5 / VOXEL_SCL;
#else
    daxa_f32vec3 ray_pos = ray_hit_ws(vrc) + nrm * 0.001;
#endif

    daxa_f32vec2 blue_noise = texelFetch(daxa_texture3D(blue_noise_vec2), ivec3(gl_GlobalInvocationID.xy, deref(gpu_input).frame_index) & ivec3(127, 127, 63), 0).yz * 255.0 / 256.0 + 0.5 / 256.0;

    daxa_f32vec3 ray_dir = sample_sun_direction(blue_noise);

    uint hit = 0;
    if (depth != 0.0) {
        VoxelTraceResult trace_result = voxel_trace(VoxelTraceInfo(VOXELS_BUFFER_PTRS, ray_dir, MAX_STEPS, MAX_DIST, 0.0, true), ray_pos);
        hit = uint(trace_result.dist == MAX_DIST);
    }

    imageStore(daxa_image2D(shadow_mask), ivec2(gl_GlobalInvocationID.xy), daxa_f32vec4(hit, 0, 0, 0));
}

#endif

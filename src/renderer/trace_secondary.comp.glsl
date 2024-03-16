#include <renderer/trace_secondary.inl>

#include <renderer/kajiya/inc/camera.glsl>
#include <voxels/voxels.glsl>
#include <renderer/atmosphere/sky.glsl>
#include <renderer/kajiya/inc/downscale.glsl>
#include <renderer/kajiya/inc/gbuffer.glsl>

#if TraceSecondaryComputeShader

DAXA_DECL_PUSH_CONSTANT(TraceSecondaryComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewIndex shadow_mask = push.uses.shadow_mask;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)
daxa_ImageViewIndex blue_noise_vec2 = push.uses.blue_noise_vec2;
daxa_ImageViewIndex g_buffer_image_id = push.uses.g_buffer_image_id;
daxa_ImageViewIndex depth_image_id = push.uses.depth_image_id;
daxa_ImageViewIndex particles_shadow_depth_tex = push.uses.particles_shadow_depth_tex;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    vec4 output_tex_size = vec4(deref(gpu_input).frame_dim, 0, 0);
    output_tex_size.zw = vec2(1.0, 1.0) / output_tex_size.xy;

    if (any(greaterThanEqual(gl_GlobalInvocationID.xy, uvec2(output_tex_size.xy)))) {
        return;
    }

    vec2 uv = get_uv(gl_GlobalInvocationID.xy, output_tex_size);
    float depth = texelFetch(daxa_texture2D(depth_image_id), ivec2(gl_GlobalInvocationID.xy), 0).r;
    GbufferDataPacked gbuffer_packed = GbufferDataPacked(texelFetch(daxa_utexture2D(g_buffer_image_id), ivec2(gl_GlobalInvocationID.xy), 0));
    GbufferData gbuffer = unpack(gbuffer_packed);
    vec3 nrm = gbuffer.normal;

    ViewRayContext vrc = vrc_from_uv_and_biased_depth(gpu_input, uv, depth);
    vec3 cam_dir = ray_dir_ws(vrc);
    vec3 cam_pos = ray_origin_ws(vrc);
    vec3 ray_origin = biased_secondary_ray_origin_ws_with_normal(vrc, nrm);
    vec3 ray_pos = ray_origin;

    vec2 blue_noise = texelFetch(daxa_texture3D(blue_noise_vec2), ivec3(gl_GlobalInvocationID.xy, deref(gpu_input).frame_index) & ivec3(127, 127, 63), 0).yz * 255.0 / 256.0 + 0.5 / 256.0;

    vec3 ray_dir = sample_sun_direction(gpu_input, blue_noise, true);

    uint hit = 0;
    if (depth != 0.0 && dot(nrm, ray_dir) > 0) {
        VoxelTraceResult trace_result = voxel_trace(VoxelTraceInfo(VOXELS_BUFFER_PTRS, ray_dir, MAX_STEPS, MAX_DIST, 0.0, true), ray_pos);
        hit = uint(trace_result.dist == MAX_DIST);
    }

    {
        vec4 hit_shadow_h = deref(gpu_input).ws_to_shadow * vec4(ray_origin, 1);
        vec3 hit_shadow = hit_shadow_h.xyz / hit_shadow_h.w;
        vec2 offset = vec2(0); // blue_noise.xy * (0.25 / 2048.0);
        float shadow_depth = texture(daxa_sampler2D(particles_shadow_depth_tex, g_sampler_nnc), cs_to_uv(hit_shadow.xy) + offset).r;

        const float bias = 0.001;
        const bool inside_shadow_map = all(greaterThanEqual(hit_shadow.xyz, vec3(-1, -1, 0))) && all(lessThanEqual(hit_shadow.xyz, vec3(+1, +1, +1)));

        if (inside_shadow_map && shadow_depth != 1.0) {
            float shadow_map_mask = sign(hit_shadow.z - shadow_depth + bias);
            hit *= uint(shadow_map_mask);
        }
    }

    imageStore(daxa_image2D(shadow_mask), ivec2(gl_GlobalInvocationID.xy), vec4(hit, 0, 0, 0));
}

#endif

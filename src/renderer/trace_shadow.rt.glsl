#define DAXA_RAY_TRACING 1
#extension GL_EXT_ray_tracing : enable

#include "trace_secondary.inl"
DAXA_DECL_PUSH_CONSTANT(TraceShadowRtPush, push)

#define TRACE 1
#include <renderer/rt.glsl>

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_RAYGEN || GL_COMPUTE_SHADER

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_RAYGEN
layout(location = PAYLOAD_LOC) rayPayloadEXT RayPayload prd;
#else
RayPayload prd;
#endif

#include <renderer/kajiya/inc/rt.glsl>
#include <renderer/kajiya/inc/camera.glsl>
#include <renderer/atmosphere/sky.glsl>
#include <renderer/kajiya/inc/downscale.glsl>
#include <renderer/kajiya/inc/gbuffer.glsl>

#if GL_COMPUTE_SHADER
layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;
#endif

void main() {
#if GL_RAY_GENERATION_SHADER_EXT
    uvec2 index = gl_LaunchIDEXT.xy;
#else
    uvec2 index = gl_GlobalInvocationID.xy;
#endif

    vec4 output_tex_size = vec4(deref(push.uses.gpu_input).frame_dim, 0, 0);
#if GL_COMPUTE_SHADER
    if (index.x >= output_tex_size.x || index.y >= output_tex_size.y)
        return;
#endif
    output_tex_size.zw = vec2(1.0, 1.0) / output_tex_size.xy;

    vec2 uv = get_uv(index, output_tex_size);
    float depth = texelFetch(daxa_texture2D(push.uses.depth_image_id), ivec2(index), 0).r;
    GbufferDataPacked gbuffer_packed = GbufferDataPacked(texelFetch(daxa_utexture2D(push.uses.g_buffer_image_id), ivec2(index), 0));
    GbufferData gbuffer = unpack(gbuffer_packed);
    vec3 nrm = gbuffer.normal;

    ViewRayContext vrc = vrc_from_uv_and_biased_depth(push.uses.gpu_input, uv, depth);
    vec3 cam_dir = ray_dir_ws(vrc);
    vec3 cam_pos = ray_origin_ws(vrc);
    vec3 ray_origin = biased_secondary_ray_origin_ws_with_normal(vrc, nrm);
    vec3 ray_pos = ray_origin;

    vec2 blue_noise = texelFetch(daxa_texture3D(push.uses.blue_noise_vec2), ivec3(index, deref(push.uses.gpu_input).frame_index) & ivec3(127, 127, 63), 0).yz * 255.0 / 256.0 + 0.5 / 256.0;

    vec3 ray_dir = sample_sun_direction(push.uses.gpu_input, blue_noise, true);

    // vec3 ws_abs = abs(ray_pos);
    // float level = floor(log2(max(max(ws_abs.x, ws_abs.y), max(ws_abs.z, 512 * VOXEL_SIZE / 2)) / (512 * VOXEL_SIZE / 2)));

    uint hit = 0;
    if (depth != 0.0 && dot(nrm, ray_dir) > 0) {
        const float t_min = 0.0001; // VOXEL_SIZE * pow(2, level) * 1.5;
        const float t_max = 10000.0;

        if (!rt_is_shadowed(new_ray(ray_pos, ray_dir, t_min, t_max))) {
            hit = 1;
        }
    }

    {
        vec4 hit_shadow_h = deref(push.uses.gpu_input).ws_to_shadow * vec4(ray_origin, 1);
        vec3 hit_shadow = hit_shadow_h.xyz / hit_shadow_h.w;
        vec2 offset = vec2(0);  // blue_noise.xy * (0.25 / 2048.0);
        float shadow_depth = texture(daxa_sampler2D(push.uses.particles_shadow_depth_tex, g_sampler_nnc), cs_to_uv(hit_shadow.xy) + offset).r;

        const float bias = 0.001;
        const bool inside_shadow_map = false; // all(greaterThanEqual(hit_shadow.xyz, vec3(-1, -1, 0))) && all(lessThanEqual(hit_shadow.xyz, vec3(+1, +1, +1)));

        if (inside_shadow_map && shadow_depth != 1.0) {
            float shadow_map_mask = sign(hit_shadow.z - shadow_depth + bias);
            hit *= uint(shadow_map_mask);
        }
    }

    imageStore(daxa_image2D(push.uses.shadow_mask), ivec2(index), vec4(hit, 0, 0, 0));
}
#endif

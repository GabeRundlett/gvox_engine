#define DAXA_RAY_TRACING 1
#extension GL_EXT_ray_tracing : enable

#include "trace_primary.inl"
DAXA_DECL_PUSH_CONSTANT(TracePrimaryRtPush, push)

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
#include <utilities/gpu/normal.glsl>
#include <voxels/pack_unpack.inl>

#if GL_COMPUTE_SHADER
layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;
#endif

void main() {
#if GL_RAY_GENERATION_SHADER_EXT
	uvec2 index = gl_LaunchIDEXT.xy;
    vec4 output_tex_size = vec4(deref(push.uses.gpu_input).frame_dim, 0, 0);
#else
	uvec2 index = gl_GlobalInvocationID.xy;
    vec4 output_tex_size = vec4(deref(push.uses.gpu_input).frame_dim, 0, 0);
	if (index.x >= output_tex_size.x || index.y >= output_tex_size.y)
		return;
#endif

    output_tex_size.zw = vec2(1.0, 1.0) / output_tex_size.xy;
    vec2 uv = get_uv(index, output_tex_size);

    ViewRayContext vrc = vrc_from_uv(push.uses.gpu_input, uv);

    RayDesc outgoing_ray;
    outgoing_ray.Direction = ray_dir_ws(vrc);
    outgoing_ray.Origin = ray_origin_ws(vrc);
    outgoing_ray.TMin = 0;
    outgoing_ray.TMax = 10000.0;

    GbufferRaytrace primary_hit_ = GbufferRaytrace_with_ray(outgoing_ray);
    primary_hit_ = with_cull_back_faces(primary_hit_, false);
    primary_hit_ = with_path_length(primary_hit_, 0);
    const GbufferPathVertex primary_hit = trace(primary_hit_);

    if (!primary_hit.is_hit) {
        imageStore(daxa_image2D(push.uses.depth_image_id), ivec2(index), vec4(0));
        imageStore(daxa_uimage2D(push.uses.g_buffer_image_id), ivec2(index), uvec4(0));
        return;
    }

    vec3 world_pos = outgoing_ray.Origin + prd.t * outgoing_ray.Direction;
    vec3 vel_ws = vec3(0);
    Voxel voxel = unpack_ray_payload(prd, push.uses.voxel_object_manifests);
    // voxel.albedo = vec3(1);
    // voxel.material_type = 1;
    // voxel.roughness = 0.1;

    // vec3 ws_abs = abs(world_pos);
    // float level = floor(log2(max(max(ws_abs.x, ws_abs.y), max(ws_abs.z, 256 * VOXEL_SIZE)) / (256 * VOXEL_SIZE)));
    // voxel.albedo = clamp(level / 4, 0, 1).xxx;

#if PER_VOXEL_NORMALS
    vec3 ws_nrm = voxel.normal;
#else
    vec3 ws_nrm = voxel_face_normal((floor(world_pos * VOXEL_SCL + outgoing_ray.Direction * 0.0001) + 0.5) * VOXEL_SIZE, Ray(outgoing_ray.Origin, outgoing_ray.Direction), vec3(1.0) / outgoing_ray.Direction);
#endif

    vec3 vs_nrm = (deref(push.uses.gpu_input).player.cam.world_to_view * vec4(ws_nrm, 0)).xyz;
    vec3 vs_velocity = vec3(0, 0, 0);

    // vel_ws += vec3(deref(push.uses.gpu_input).player.player_unit_offset - deref(push.uses.gpu_input).player.prev_unit_offset);

    vec4 vs_pos = (deref(push.uses.gpu_input).player.cam.world_to_view * vec4(world_pos, 1));
    vec4 prev_vs_pos = (deref(push.uses.gpu_input).player.cam.world_to_view * vec4(world_pos + vel_ws, 1));
    vec4 ss_pos = (deref(push.uses.gpu_input).player.cam.view_to_sample * vs_pos);
    float depth = ss_pos.z / ss_pos.w;

    vs_velocity = (prev_vs_pos.xyz / prev_vs_pos.w) - (vs_pos.xyz / vs_pos.w);

    uvec4 output_value = uvec4(0);
    output_value.x = pack_voxel(voxel).data;
    output_value.y = nrm_to_u16(ws_nrm);
    output_value.z = floatBitsToUint(depth);

    vs_nrm *= -sign(dot(ray_dir_vs(vrc), vs_nrm));

    imageStore(daxa_uimage2D(push.uses.g_buffer_image_id), ivec2(index), output_value);
    imageStore(daxa_image2D(push.uses.velocity_image_id), ivec2(index), vec4(vs_velocity, 0));
    imageStore(daxa_image2D(push.uses.vs_normal_image_id), ivec2(index), vec4(vs_nrm * 0.5 + 0.5, 0));
    imageStore(daxa_image2D(push.uses.depth_image_id), ivec2(index), vec4(depth, 0, 0, 0));
}
#endif

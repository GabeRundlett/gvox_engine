#define DAXA_RAY_TRACING 1
#extension GL_EXT_ray_tracing : enable

#include "trace_primary.inl"
DAXA_DECL_PUSH_CONSTANT(TracePrimaryRtPush, push)
#include <renderer/rt.glsl>

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_RAYGEN

layout(location = PAYLOAD_LOC) rayPayloadEXT RayPayload prd;

#include <renderer/kajiya/inc/camera.glsl>
#include <utilities/gpu/normal.glsl>
#include <voxels/pack_unpack.inl>

void main() {
    const ivec2 index = ivec2(gl_LaunchIDEXT.xy);

    vec4 output_tex_size = vec4(deref(push.uses.gpu_input).frame_dim, 0, 0);
    output_tex_size.zw = vec2(1.0, 1.0) / output_tex_size.xy;
    vec2 uv = get_uv(gl_LaunchIDEXT.xy, output_tex_size);

    ViewRayContext vrc = vrc_from_uv(push.uses.gpu_input, uv);
    vec3 ray_d = ray_dir_ws(vrc);
    vec3 ray_o = ray_origin_ws(vrc);

    const uint ray_flags = gl_RayFlagsNoneEXT;
    const uint cull_mask = 0xFF & ~(0x01 & ~(deref(push.uses.gpu_input).player.flags & 1));
    const uint sbt_record_offset = 0;
    const uint sbt_record_stride = 0;
    const uint miss_index = 0;
    const float t_min = 0.0001;
    const float t_max = 10000.0;

    traceRayEXT(
        accelerationStructureEXT(push.uses.tlas),
        ray_flags, cull_mask, sbt_record_offset, sbt_record_stride, miss_index,
        ray_o, t_min, ray_d, t_max, PAYLOAD_LOC);

    if (prd.data1 == miss_ray_payload().data1) {
        imageStore(daxa_image2D(push.uses.depth_image_id), ivec2(gl_LaunchIDEXT.xy), vec4(0));
        imageStore(daxa_uimage2D(push.uses.g_buffer_image_id), ivec2(gl_LaunchIDEXT.xy), uvec4(0));
        return;
    }

    vec3 world_pos = ray_o + prd.t * ray_d;
    vec3 vel_ws = vec3(0);
    GpuVoxel voxel = unpack_ray_payload(prd, push.uses.voxel_object_manifests);
    // voxel.albedo = vec3(0.01);

    // voxel.roughness = 0.01;

#if PER_VOXEL_NORMALS
    vec3 ws_nrm = voxel.normal;
#else
    vec3 ws_nrm = voxel_face_normal((floor(world_pos * VOXEL_SCL + ray_d * 0.0001) + 0.5) * VOXEL_SIZE, Ray(ray_o, ray_d), vec3(1.0) / ray_d);
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

    imageStore(daxa_uimage2D(push.uses.g_buffer_image_id), ivec2(gl_LaunchIDEXT.xy), output_value);
    imageStore(daxa_image2D(push.uses.velocity_image_id), ivec2(gl_LaunchIDEXT.xy), vec4(vs_velocity, 0));
    imageStore(daxa_image2D(push.uses.vs_normal_image_id), ivec2(gl_LaunchIDEXT.xy), vec4(vs_nrm * 0.5 + 0.5, 0));
    imageStore(daxa_image2D(push.uses.depth_image_id), ivec2(gl_LaunchIDEXT.xy), vec4(depth, 0, 0, 0));
}
#endif

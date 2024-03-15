#include "voxel_particles.inl"

#include <utilities/gpu/defs.glsl>
#include <renderer/kajiya/inc/camera.glsl>

#if defined(GRASS)
DAXA_DECL_PUSH_CONSTANT(GrassStrandSplatParticleRasterPush, push)
daxa_BufferPtr(GrassStrand) grass_strands = push.uses.grass_strands;
#else
DAXA_DECL_PUSH_CONSTANT(SimParticleSplatParticleRasterPush, push)
daxa_BufferPtr(SimulatedVoxelParticle) simulated_voxel_particles = push.uses.simulated_voxel_particles;
#endif
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(ParticleVertex) splat_rendered_particle_verts = push.uses.splat_rendered_particle_verts;

#define SHADING
#include "particle.glsl"

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

layout(location = 0) out vec3 center_ws;
layout(location = 1) out uint id;

void main() {
    uint particle_index = gl_VertexIndex;

    ParticleVertex particle = deref(advance(splat_rendered_particle_verts, particle_index));

    float voxel_radius = VOXEL_SIZE * 0.5;
    center_ws = get_particle_worldspace_origin(gpu_input, particle.pos);
    mat4 world_to_sample = deref(gpu_input).player.cam.view_to_sample * deref(gpu_input).player.cam.world_to_view;
    vec2 half_screen_size = vec2(deref(gpu_input).frame_dim) * 0.5;
    float ps_size = 0.0;
    vec2 px_pos = vec2(0, 0);
    particle_point_pos_and_size(center_ws, voxel_radius, world_to_sample, half_screen_size, px_pos, ps_size);

    vec4 vs_pos = deref(gpu_input).player.cam.world_to_view * vec4(center_ws, 1);
    vec4 cs_pos = deref(gpu_input).player.cam.view_to_sample * vs_pos;
    cs_pos.y *= -1;

    gl_Position = cs_pos;
    gl_PointSize = ps_size;

    id = particle.id;
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT

layout(location = 0) flat in vec3 center_ws;
layout(location = 1) flat in uint id;

layout(location = 0) out uvec4 o_gbuffer;
layout(location = 1) out vec4 o_vs_velocity;
layout(location = 2) out vec4 o_vs_nrm;

void main() {
    Box box;
    box.radius = vec3(VOXEL_SIZE * 0.5);
    box.invRadius = 1.0 / box.radius;
    box.center = center_ws;

    vec4 output_tex_size = vec4(deref(gpu_input).frame_dim, 0, 0);
    output_tex_size.zw = vec2(1.0, 1.0) / output_tex_size.xy;
    vec2 uv = get_uv(gl_FragCoord.xy, output_tex_size);
    ViewRayContext vrc = vrc_from_uv(gpu_input, uv);

    Ray ray;
    ray.origin = ray_origin_ws(vrc);
    ray.direction = ray_dir_ws(vrc);

    float dist;
    vec3 temp_nrm;

    if (!ourIntersectBox(box, ray, dist, temp_nrm, false, 1.0 / ray.direction)) {
        discard;
    }

    vec3 ws_pos = ray.origin + ray.direction * dist;
    vec4 vs_pos = deref(gpu_input).player.cam.world_to_view * vec4(ws_pos, 1);
    vec4 cs_pos = deref(gpu_input).player.cam.view_to_sample * vs_pos;
    float ndc_depth = cs_pos.z / cs_pos.w;

    gl_FragDepth = ndc_depth;

    vec3 nrm;
    vec3 vs_velocity;
    ParticleVertex particle_vert = ParticleVertex(center_ws, id);
    uint gbuffer_x;
    particle_shade(vrc, gpu_input, particle_vert, gbuffer_x, nrm, vs_velocity);

#if !PER_VOXEL_NORMALS
    // TODO: Fix the face-normals for splatted particles. At a distance, this ourIntersectBox function appears to return mush.
    nrm = temp_nrm;
#endif

    vec3 vs_nrm = (deref(gpu_input).player.cam.world_to_view * vec4(nrm, 0)).xyz;
    o_gbuffer.x = gbuffer_x;
    o_gbuffer.y = nrm_to_u16(nrm);
    o_gbuffer.z = floatBitsToUint(gl_FragDepth);
    vs_nrm *= -sign(dot(ray_dir_vs(vrc), vs_nrm));
    o_vs_velocity = vec4(vs_velocity, 0);
    o_vs_nrm = vec4(vs_nrm * 0.5 + 0.5, 0);
}

#endif

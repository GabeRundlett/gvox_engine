#include "voxel_particles.inl"

#include <utilities/gpu/defs.glsl>

#if defined(SHADOW_MAP)
DAXA_DECL_PUSH_CONSTANT(SimParticleCubeParticleRasterShadowPush, push)
#else

#if defined(GRASS)
DAXA_DECL_PUSH_CONSTANT(GrassStrandCubeParticleRasterPush, push)
daxa_BufferPtr(GrassStrand) grass_strands = push.uses.grass_strands;
#elif defined(SIM_PARTICLE)
DAXA_DECL_PUSH_CONSTANT(SimParticleCubeParticleRasterPush, push)
daxa_BufferPtr(SimulatedVoxelParticle) simulated_voxel_particles = push.uses.simulated_voxel_particles;
#endif
#define SHADING

#endif

daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(ParticleVertex) cube_rendered_particle_verts = push.uses.cube_rendered_particle_verts;

#include "particle.glsl"

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

#include <renderer/kajiya/inc/camera.glsl>

#if !defined(SHADOW_MAP)
layout(location = 0) out vec3 center_ws;
layout(location = 1) out uint id;
layout(location = 2) out vec3 i_nrm;
#endif

void main() {
    uint particle_index = gl_InstanceIndex;

    ParticleVertex particle = deref(advance(cube_rendered_particle_verts, particle_index));

    const vec3 diff = vec3(VOXEL_SIZE);
#if !defined(SHADOW_MAP)
    center_ws = get_particle_worldspace_origin(gpu_input, particle.pos);
#else
    vec3 center_ws = get_particle_worldspace_origin(gpu_input, particle.pos);
#endif
    const vec3 camera_position = deref(gpu_input).player.pos;

    // extracting the vertex offset relative to the center.
    // Thanks to @cantaslaus on Discord.
    vec3 sign_ = vec3(ivec3(greaterThan(camera_position, center_ws)) ^ ((ivec3(0x1C, 0x46, 0x70) >> gl_VertexIndex) & ivec3(1))) - 0.5;
    vec3 vert_pos = sign_ * diff + center_ws;
    // ---------------------
    vec3 nrm = vec3((ivec3(greaterThan(camera_position, center_ws)) * 2 - 1) * ((ivec3(0x60, 0x18, 0x06) >> gl_VertexIndex) & 1));

#if defined(SHADOW_MAP)
    vec4 cs_pos = deref(gpu_input).ws_to_shadow * vec4(vert_pos, 1);
#else
    vec4 vs_pos = deref(gpu_input).player.cam.world_to_view * vec4(vert_pos, 1);
    vec4 cs_pos = deref(gpu_input).player.cam.view_to_sample * vs_pos;
    id = particle.id;
    i_nrm = nrm;
#endif

    // TODO: Figure out why raster is projected upside down
    cs_pos.y *= -1;
    gl_Position = cs_pos;
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT

#if defined(SHADOW_MAP)
void main() {
}
#else
#include <renderer/kajiya/inc/camera.glsl>

layout(location = 0) flat in vec3 center_ws;
layout(location = 1) flat in uint id;
layout(location = 2) flat in vec3 i_nrm;

layout(location = 0) out uvec4 o_gbuffer;
layout(location = 1) out vec4 o_vs_velocity;
layout(location = 2) out vec4 o_vs_nrm;

void main() {
    vec4 output_tex_size = vec4(deref(gpu_input).frame_dim, 0, 0);
    output_tex_size.zw = vec2(1.0, 1.0) / output_tex_size.xy;
    vec2 uv = get_uv(gl_FragCoord.xy, output_tex_size);
    ViewRayContext vrc = vrc_from_uv(gpu_input, uv);

    vec3 nrm;
    vec3 vs_velocity;
    ParticleVertex particle_vert = ParticleVertex(center_ws, id);
    uint gbuffer_x;
    particle_shade(vrc, gpu_input, particle_vert, gbuffer_x, nrm, vs_velocity);
#if !PER_VOXEL_NORMALS
    nrm = i_nrm;
#endif
    vec3 vs_nrm = (deref(gpu_input).player.cam.world_to_view * vec4(nrm, 0)).xyz;
    o_gbuffer.x = gbuffer_x;
    o_gbuffer.y = nrm_to_u16(nrm);
    o_gbuffer.z = floatBitsToUint(gl_FragDepth);
    // vs_nrm *= -sign(dot(ray_dir_vs(vrc), vs_nrm));
    o_vs_velocity = vec4(vs_velocity, 0);
    o_vs_nrm = vec4(vs_nrm * 0.5 + 0.5, 0);
}
#endif

#endif

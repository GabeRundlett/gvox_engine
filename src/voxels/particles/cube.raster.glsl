#include "voxel_particles.inl"

#include <utilities/gpu/defs.glsl>

#if defined(SHADOW_MAP)

#if defined(GRASS)
DAXA_DECL_PUSH_CONSTANT(GrassStrandCubeParticleRasterShadowPush, push)
daxa_BufferPtr(GrassStrand) grass_strands = push.uses.grass_strands;
#elif defined(FLOWER)
DAXA_DECL_PUSH_CONSTANT(FlowerCubeParticleRasterShadowPush, push)
daxa_BufferPtr(Flower) flowers = push.uses.flowers;
#elif defined(SIM_PARTICLE)
DAXA_DECL_PUSH_CONSTANT(SimParticleCubeParticleRasterShadowPush, push)
daxa_BufferPtr(SimulatedVoxelParticle) simulated_voxel_particles = push.uses.simulated_voxel_particles;
#elif defined(TREE_PARTICLE)
DAXA_DECL_PUSH_CONSTANT(TreeParticleCubeParticleRasterShadowPush, push)
daxa_BufferPtr(TreeParticle) tree_particles = push.uses.tree_particles;
#endif

#else

#if defined(GRASS)
DAXA_DECL_PUSH_CONSTANT(GrassStrandCubeParticleRasterPush, push)
daxa_BufferPtr(GrassStrand) grass_strands = push.uses.grass_strands;
#elif defined(FLOWER)
DAXA_DECL_PUSH_CONSTANT(FlowerCubeParticleRasterPush, push)
daxa_BufferPtr(Flower) flowers = push.uses.flowers;
#elif defined(SIM_PARTICLE)
DAXA_DECL_PUSH_CONSTANT(SimParticleCubeParticleRasterPush, push)
daxa_BufferPtr(SimulatedVoxelParticle) simulated_voxel_particles = push.uses.simulated_voxel_particles;
#elif defined(TREE_PARTICLE)
DAXA_DECL_PUSH_CONSTANT(TreeParticleCubeParticleRasterPush, push)
daxa_BufferPtr(TreeParticle) tree_particles = push.uses.tree_particles;
#endif

#endif

daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(PackedParticleVertex) cube_rendered_particle_verts = push.uses.cube_rendered_particle_verts;

#include "grass/grass.glsl"
#include "flower/flower.glsl"
#include "sim_particle/sim_particle.glsl"
#include "tree_particle/tree_particle.glsl"

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

#include <renderer/kajiya/inc/camera.glsl>

#if !defined(SHADOW_MAP)
layout(location = 0) out uvec2 i_gbuffer_xy;
layout(location = 1) out vec3 i_vs_nrm;
layout(location = 2) out vec3 i_vs_velocity;
#endif

void main() {
    uint particle_index = gl_InstanceIndex;

    PackedParticleVertex packed_vertex = deref(advance(cube_rendered_particle_verts, particle_index));

#if defined(GRASS)
    ParticleVertex vert = get_grass_vertex(gpu_input, grass_strands, packed_vertex);
#elif defined(FLOWER)
    ParticleVertex vert = get_flower_vertex(gpu_input, flowers, packed_vertex);
#elif defined(SIM_PARTICLE)
    ParticleVertex vert = get_sim_particle_vertex(gpu_input, simulated_voxel_particles, packed_vertex);
#elif defined(TREE_PARTICLE)
    ParticleVertex vert = get_tree_particle_vertex(gpu_input, tree_particles, packed_vertex);
#endif

    const vec3 diff = vec3(VOXEL_SIZE);
    vec3 center_ws = vert.pos;
    const vec3 camera_position = deref(gpu_input).player.pos;
    const vec3 camera_to_center = center_ws - camera_position;
    const vec3 ray_dir_ws = normalize(camera_to_center);
    const vec3 ray_dir_vs = (deref(gpu_input).player.cam.world_to_view * vec4(ray_dir_ws, 0)).xyz;

    // extracting the vertex offset relative to the center.
    // Thanks to @cantaslaus on Discord.
    vec3 sign_ = vec3(ivec3(greaterThan(camera_position, center_ws)) ^ ((ivec3(0x1C, 0x46, 0x70) >> gl_VertexIndex) & ivec3(1))) - 0.5;
    vec3 vert_pos = sign_ * diff + center_ws;
    // ---------------------

#if defined(SHADOW_MAP)
    vec4 cs_pos = deref(gpu_input).ws_to_shadow * vec4(vert_pos, 1);
#else
    vec4 vs_pos = (deref(gpu_input).player.cam.world_to_view * vec4(vert.pos, 1));
    vec4 prev_vs_pos = (deref(gpu_input).player.cam.world_to_view * vec4(vert.prev_pos, 1));
    i_vs_velocity = (prev_vs_pos.xyz / prev_vs_pos.w) - (vs_pos.xyz / vs_pos.w);
    Voxel voxel = unpack_voxel(vert.packed_voxel);
    vec3 nrm = voxel.normal;
#if !PER_VOXEL_NORMALS
    vec3 face_nrm = vec3((ivec3(greaterThan(camera_position, center_ws)) * 2 - 1) * ((ivec3(0x60, 0x18, 0x06) >> gl_VertexIndex) & 1));
    nrm = face_nrm;
#endif
    i_gbuffer_xy = uvec2(vert.packed_voxel.data, nrm_to_u16(nrm));
    vec3 vs_nrm = (deref(gpu_input).player.cam.world_to_view * vec4(nrm, 0)).xyz;
    i_vs_nrm = vs_nrm * -sign(dot(ray_dir_vs, vs_nrm));
    vec4 vs_pos2 = deref(gpu_input).player.cam.world_to_view * vec4(vert_pos, 1);
    vec4 cs_pos = deref(gpu_input).player.cam.view_to_sample * vs_pos2;
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

layout(location = 0) flat in uvec2 i_gbuffer_xy;
layout(location = 1) flat in vec3 i_vs_nrm;
layout(location = 2) flat in vec3 i_vs_velocity;

layout(location = 0) out uvec4 o_gbuffer;
layout(location = 1) out vec4 o_vs_velocity;
layout(location = 2) out vec4 o_vs_nrm;

void main() {
    o_gbuffer = uvec4(i_gbuffer_xy, floatBitsToUint(gl_FragCoord.z), 0);
    o_vs_velocity = vec4(i_vs_velocity, 0);
    o_vs_nrm = vec4(i_vs_nrm * 0.5 + 0.5, 0);
}
#endif

#endif

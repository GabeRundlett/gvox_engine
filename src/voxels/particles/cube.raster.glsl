#include "voxel_particles.inl"

#include <utilities/gpu/defs.glsl>

DAXA_DECL_PUSH_CONSTANT(VoxelParticleRasterPush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(SimulatedVoxelParticle) simulated_voxel_particles = push.uses.simulated_voxel_particles;
daxa_BufferPtr(uint) rendered_voxel_particles = push.uses.rendered_voxel_particles;
daxa_ImageViewIndex render_image = push.uses.render_image;
daxa_ImageViewIndex depth_image_id = push.uses.depth_image_id;

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

#include "particle.glsl"
#include <renderer/kajiya/inc/camera.glsl>

layout(location = 1) out uint id;

void main() {
    uint particle_index = gl_VertexIndex / 36;
    vec3 positions[36];

    positions[0 + 6 * 0] = vec3(+0.0, +0.0, +0.0);
    positions[1 + 6 * 0] = vec3(+1.0, +0.0, +0.0);
    positions[2 + 6 * 0] = vec3(+0.0, +1.0, +0.0);
    positions[3 + 6 * 0] = vec3(+0.0, +1.0, +0.0);
    positions[4 + 6 * 0] = vec3(+1.0, +0.0, +0.0);
    positions[5 + 6 * 0] = vec3(+1.0, +1.0, +0.0);

    positions[0 + 6 * 1] = vec3(+0.0, +0.0, +1.0);
    positions[1 + 6 * 1] = vec3(+0.0, +1.0, +1.0);
    positions[2 + 6 * 1] = vec3(+1.0, +0.0, +1.0);
    positions[3 + 6 * 1] = vec3(+1.0, +0.0, +1.0);
    positions[4 + 6 * 1] = vec3(+0.0, +1.0, +1.0);
    positions[5 + 6 * 1] = vec3(+1.0, +1.0, +1.0);

    positions[0 + 6 * 2] = vec3(+0.0, +0.0, +0.0);
    positions[1 + 6 * 2] = vec3(+0.0, +0.0, +1.0);
    positions[2 + 6 * 2] = vec3(+1.0, +0.0, +0.0);
    positions[3 + 6 * 2] = vec3(+1.0, +0.0, +0.0);
    positions[4 + 6 * 2] = vec3(+0.0, +0.0, +1.0);
    positions[5 + 6 * 2] = vec3(+1.0, +0.0, +1.0);

    positions[0 + 6 * 3] = vec3(+0.0, +1.0, +0.0);
    positions[1 + 6 * 3] = vec3(+1.0, +1.0, +0.0);
    positions[2 + 6 * 3] = vec3(+0.0, +1.0, +1.0);
    positions[3 + 6 * 3] = vec3(+0.0, +1.0, +1.0);
    positions[4 + 6 * 3] = vec3(+1.0, +1.0, +0.0);
    positions[5 + 6 * 3] = vec3(+1.0, +1.0, +1.0);

    positions[0 + 6 * 4] = vec3(+0.0, +0.0, +0.0);
    positions[1 + 6 * 4] = vec3(+0.0, +1.0, +0.0);
    positions[2 + 6 * 4] = vec3(+0.0, +0.0, +1.0);
    positions[3 + 6 * 4] = vec3(+0.0, +0.0, +1.0);
    positions[4 + 6 * 4] = vec3(+0.0, +1.0, +0.0);
    positions[5 + 6 * 4] = vec3(+0.0, +1.0, +1.0);

    positions[0 + 6 * 5] = vec3(+1.0, +0.0, +0.0);
    positions[1 + 6 * 5] = vec3(+1.0, +0.0, +1.0);
    positions[2 + 6 * 5] = vec3(+1.0, +1.0, +0.0);
    positions[3 + 6 * 5] = vec3(+1.0, +1.0, +0.0);
    positions[4 + 6 * 5] = vec3(+1.0, +0.0, +1.0);
    positions[5 + 6 * 5] = vec3(+1.0, +1.0, +1.0);

    uint simulated_particle_index = deref(advance(rendered_voxel_particles, particle_index));
    SimulatedVoxelParticle particle = deref(advance(simulated_voxel_particles, simulated_particle_index));

    vec3 particle_worldspace_origin = get_particle_worldspace_origin(particle.pos);
    vec3 cube_voxel_vertex = (positions[gl_VertexIndex - particle_index * 36] * (1023.0 / 1024.0) + (1.0 / 2048.0)) * VOXEL_SIZE;
    vec3 vert_pos = cube_voxel_vertex + particle_worldspace_origin;

    vec4 vs_pos = deref(gpu_input).player.cam.world_to_view * vec4(vert_pos, 1);
    vec4 cs_pos = deref(gpu_input).player.cam.view_to_sample * vs_pos;
    // TODO: Figure out why raster is projected upside down
    cs_pos.y *= -1;

    gl_Position = cs_pos;

    id = simulated_particle_index;
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT

layout(location = 1) flat in uint id;
layout(location = 0) out uvec4 color;

void main() {
    color = uvec4(id + 1, 0, 0, 0);
}

#endif

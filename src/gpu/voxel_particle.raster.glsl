#include <shared/shared.inl>
#define VOXEL_SCL 8

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

layout(location = 0) out f32vec3 pos;
layout(location = 1) out uint id;

void main() {
    u32 particle_index = gl_VertexIndex / 36;
    f32vec3 positions[36];

    positions[0 + 6 * 0] = f32vec3(+0.0, +0.0, +0.0);
    positions[1 + 6 * 0] = f32vec3(+1.0, +0.0, +0.0);
    positions[2 + 6 * 0] = f32vec3(+0.0, +1.0, +0.0);
    positions[3 + 6 * 0] = f32vec3(+0.0, +1.0, +0.0);
    positions[4 + 6 * 0] = f32vec3(+1.0, +0.0, +0.0);
    positions[5 + 6 * 0] = f32vec3(+1.0, +1.0, +0.0);

    positions[0 + 6 * 1] = f32vec3(+0.0, +0.0, +1.0);
    positions[1 + 6 * 1] = f32vec3(+0.0, +1.0, +1.0);
    positions[2 + 6 * 1] = f32vec3(+1.0, +0.0, +1.0);
    positions[3 + 6 * 1] = f32vec3(+1.0, +0.0, +1.0);
    positions[4 + 6 * 1] = f32vec3(+0.0, +1.0, +1.0);
    positions[5 + 6 * 1] = f32vec3(+1.0, +1.0, +1.0);

    positions[0 + 6 * 2] = f32vec3(+0.0, +0.0, +0.0);
    positions[1 + 6 * 2] = f32vec3(+0.0, +0.0, +1.0);
    positions[2 + 6 * 2] = f32vec3(+1.0, +0.0, +0.0);
    positions[3 + 6 * 2] = f32vec3(+1.0, +0.0, +0.0);
    positions[4 + 6 * 2] = f32vec3(+0.0, +0.0, +1.0);
    positions[5 + 6 * 2] = f32vec3(+1.0, +0.0, +1.0);

    positions[0 + 6 * 3] = f32vec3(+0.0, +1.0, +0.0);
    positions[1 + 6 * 3] = f32vec3(+1.0, +1.0, +0.0);
    positions[2 + 6 * 3] = f32vec3(+0.0, +1.0, +1.0);
    positions[3 + 6 * 3] = f32vec3(+0.0, +1.0, +1.0);
    positions[4 + 6 * 3] = f32vec3(+1.0, +1.0, +0.0);
    positions[5 + 6 * 3] = f32vec3(+1.0, +1.0, +1.0);

    positions[0 + 6 * 4] = f32vec3(+0.0, +0.0, +0.0);
    positions[1 + 6 * 4] = f32vec3(+0.0, +1.0, +0.0);
    positions[2 + 6 * 4] = f32vec3(+0.0, +0.0, +1.0);
    positions[3 + 6 * 4] = f32vec3(+0.0, +0.0, +1.0);
    positions[4 + 6 * 4] = f32vec3(+0.0, +1.0, +0.0);
    positions[5 + 6 * 4] = f32vec3(+0.0, +1.0, +1.0);

    positions[0 + 6 * 5] = f32vec3(+1.0, +0.0, +0.0);
    positions[1 + 6 * 5] = f32vec3(+1.0, +0.0, +1.0);
    positions[2 + 6 * 5] = f32vec3(+1.0, +1.0, +0.0);
    positions[3 + 6 * 5] = f32vec3(+1.0, +1.0, +0.0);
    positions[4 + 6 * 5] = f32vec3(+1.0, +0.0, +1.0);
    positions[5 + 6 * 5] = f32vec3(+1.0, +1.0, +1.0);

    u32 simulated_particle_index = deref(rendered_voxel_particles[particle_index]);
    SimulatedVoxelParticle particle = deref(simulated_voxel_particles[simulated_particle_index]);

    f32mat4x4 vp_mat = deref(globals).player.cam.proj_mat;
    vec3 vert_pos = (positions[gl_VertexIndex - particle_index * 36] * (1023.0 / 1024.0) + (1.0 / 2048.0)) / VOXEL_SCL + floor(particle.pos * VOXEL_SCL) / VOXEL_SCL;

    pos = vert_pos;
    gl_Position = vp_mat * f32vec4(vert_pos, 1);

    id = simulated_particle_index;
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT

layout(location = 0) in f32vec3 pos;
layout(location = 1) flat in uint id;
layout(location = 0) out f32vec4 color;

void main() {
    color = f32vec4(pos, 1 + id);
}

#endif

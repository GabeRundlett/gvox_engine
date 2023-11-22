#include <shared/app.inl>
#define VOXEL_SCL 8

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

layout(location = 0) out daxa_f32vec3 pos;
layout(location = 1) out uint id;

void main() {
    daxa_u32 particle_index = gl_VertexIndex / 36;
    daxa_f32vec3 positions[36];

    positions[0 + 6 * 0] = daxa_f32vec3(+0.0, +0.0, +0.0);
    positions[1 + 6 * 0] = daxa_f32vec3(+1.0, +0.0, +0.0);
    positions[2 + 6 * 0] = daxa_f32vec3(+0.0, +1.0, +0.0);
    positions[3 + 6 * 0] = daxa_f32vec3(+0.0, +1.0, +0.0);
    positions[4 + 6 * 0] = daxa_f32vec3(+1.0, +0.0, +0.0);
    positions[5 + 6 * 0] = daxa_f32vec3(+1.0, +1.0, +0.0);

    positions[0 + 6 * 1] = daxa_f32vec3(+0.0, +0.0, +1.0);
    positions[1 + 6 * 1] = daxa_f32vec3(+0.0, +1.0, +1.0);
    positions[2 + 6 * 1] = daxa_f32vec3(+1.0, +0.0, +1.0);
    positions[3 + 6 * 1] = daxa_f32vec3(+1.0, +0.0, +1.0);
    positions[4 + 6 * 1] = daxa_f32vec3(+0.0, +1.0, +1.0);
    positions[5 + 6 * 1] = daxa_f32vec3(+1.0, +1.0, +1.0);

    positions[0 + 6 * 2] = daxa_f32vec3(+0.0, +0.0, +0.0);
    positions[1 + 6 * 2] = daxa_f32vec3(+0.0, +0.0, +1.0);
    positions[2 + 6 * 2] = daxa_f32vec3(+1.0, +0.0, +0.0);
    positions[3 + 6 * 2] = daxa_f32vec3(+1.0, +0.0, +0.0);
    positions[4 + 6 * 2] = daxa_f32vec3(+0.0, +0.0, +1.0);
    positions[5 + 6 * 2] = daxa_f32vec3(+1.0, +0.0, +1.0);

    positions[0 + 6 * 3] = daxa_f32vec3(+0.0, +1.0, +0.0);
    positions[1 + 6 * 3] = daxa_f32vec3(+1.0, +1.0, +0.0);
    positions[2 + 6 * 3] = daxa_f32vec3(+0.0, +1.0, +1.0);
    positions[3 + 6 * 3] = daxa_f32vec3(+0.0, +1.0, +1.0);
    positions[4 + 6 * 3] = daxa_f32vec3(+1.0, +1.0, +0.0);
    positions[5 + 6 * 3] = daxa_f32vec3(+1.0, +1.0, +1.0);

    positions[0 + 6 * 4] = daxa_f32vec3(+0.0, +0.0, +0.0);
    positions[1 + 6 * 4] = daxa_f32vec3(+0.0, +1.0, +0.0);
    positions[2 + 6 * 4] = daxa_f32vec3(+0.0, +0.0, +1.0);
    positions[3 + 6 * 4] = daxa_f32vec3(+0.0, +0.0, +1.0);
    positions[4 + 6 * 4] = daxa_f32vec3(+0.0, +1.0, +0.0);
    positions[5 + 6 * 4] = daxa_f32vec3(+0.0, +1.0, +1.0);

    positions[0 + 6 * 5] = daxa_f32vec3(+1.0, +0.0, +0.0);
    positions[1 + 6 * 5] = daxa_f32vec3(+1.0, +0.0, +1.0);
    positions[2 + 6 * 5] = daxa_f32vec3(+1.0, +1.0, +0.0);
    positions[3 + 6 * 5] = daxa_f32vec3(+1.0, +1.0, +0.0);
    positions[4 + 6 * 5] = daxa_f32vec3(+1.0, +0.0, +1.0);
    positions[5 + 6 * 5] = daxa_f32vec3(+1.0, +1.0, +1.0);

    daxa_u32 simulated_particle_index = deref(rendered_voxel_particles[particle_index]);
    SimulatedVoxelParticle particle = deref(simulated_voxel_particles[simulated_particle_index]);

    vec3 vert_pos = (positions[gl_VertexIndex - particle_index * 36] * (1023.0 / 1024.0) + (1.0 / 2048.0)) / VOXEL_SCL + floor(particle.pos * VOXEL_SCL) / VOXEL_SCL;

    pos = vert_pos;
    vec4 vs_pos = deref(globals).player.cam.world_to_view * daxa_f32vec4(vert_pos, 1);
    vec4 cs_pos = deref(globals).player.cam.view_to_clip * vs_pos;

    gl_Position = cs_pos;

    id = simulated_particle_index;
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT

layout(location = 0) in daxa_f32vec3 pos;
layout(location = 1) flat in uint id;
layout(location = 0) out daxa_f32vec4 color;

void main() {
    // color = daxa_f32vec4(pos, uintBitsToFloat(1 + id));
    color = daxa_f32vec4(pos, float(1 + id));
}

#endif

#include <shared/app.inl>

#define VOXEL_SCL 8

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

#include <voxels/voxel_particle.glsl>
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

    vec3 particle_worldspace_origin = get_particle_worldspace_origin(globals, particle.pos);
    vec3 cube_voxel_vertex = (positions[gl_VertexIndex - particle_index * 36] * (1023.0 / 1024.0) + (1.0 / 2048.0)) / VOXEL_SCL;
    vec3 vert_pos = cube_voxel_vertex + particle_worldspace_origin;

    mat4 view_to_clip = deref(globals).player.cam.view_to_clip;
    // TODO: Figure out why raster is projected upside down
    view_to_clip[1][1] *= -1.0;
    
    daxa_f32vec4 output_tex_size;
    output_tex_size.xy = deref(gpu_input).frame_dim;
    output_tex_size.zw = daxa_f32vec2(1.0, 1.0) / output_tex_size.xy;
    daxa_f32mat4x4 inv_jitter_mat = daxa_f32mat4x4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        ss_to_uv(gpu_input, daxa_f32vec2(0.0), output_tex_size), 0, 1);

    view_to_clip = inv_jitter_mat * view_to_clip;

    vec4 vs_pos = deref(globals).player.cam.world_to_view * daxa_f32vec4(vert_pos, 1);
    vec4 cs_pos = view_to_clip * vs_pos;

    gl_Position = cs_pos;

    id = simulated_particle_index;
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT

layout(location = 1) flat in uint id;
layout(location = 0) out daxa_u32vec4 color;

void main() {
    color = daxa_u32vec4(id + 1, 0, 0, 0);
}

#endif

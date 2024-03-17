#pragma once

#include <voxels/particles/voxel_particles.inl>
#include <utilities/gpu/math.glsl>
#include <voxels/voxels.glsl>
#include <renderer/kajiya/inc/camera.glsl>

#define PARTICLE_ALIVE_FLAG (1 << 0)
#define PARTICLE_SMOKE_FLAG (1 << 1)

#define PARTICLE_SLEEP_TIMER_OFFSET (8)
#define PARTICLE_SLEEP_TIMER_MASK (0xff << PARTICLE_SLEEP_TIMER_OFFSET)

vec3 get_particle_pos(vec3 pos) {
    // return pos;
    return floor(pos * VOXEL_SCL) * VOXEL_SIZE;
}

vec3 get_particle_worldspace_origin(daxa_BufferPtr(GpuInput) gpu_input, vec3 pos) {
    return get_particle_pos(pos) - deref(gpu_input).player.player_unit_offset + 0.5 * VOXEL_SIZE;
}

vec3 get_particle_prev_worldspace_origin(daxa_BufferPtr(GpuInput) gpu_input, vec3 pos) {
    return get_particle_pos(pos) - deref(gpu_input).player.prev_unit_offset + 0.5 * VOXEL_SIZE;
}

vec3 get_particle_worldspace_pos(daxa_BufferPtr(GpuInput) gpu_input, vec3 pos) {
    return pos - deref(gpu_input).player.player_unit_offset;
}

void particle_point_pos_and_size(in vec3 osPosition, in float voxelSize, in mat4 objectToScreenMatrix, in vec2 halfScreenSize, out vec2 px_pos, inout float pointSize) {
    const vec4 quadricMat = vec4(1.0, 1.0, 1.0, -1.0);
    float sphereRadius = voxelSize * 1.732051;
    vec4 sphereCenter = vec4(osPosition.xyz, 1.0);
    mat4 modelViewProj = transpose(objectToScreenMatrix);
    mat3x4 matT = mat3x4(mat3(modelViewProj[0].xyz, modelViewProj[1].xyz, modelViewProj[3].xyz) * sphereRadius);
    matT[0].w = dot(sphereCenter, modelViewProj[0]);
    matT[1].w = dot(sphereCenter, modelViewProj[1]);
    matT[2].w = dot(sphereCenter, modelViewProj[3]);
    mat3x4 matD = mat3x4(matT[0] * quadricMat, matT[1] * quadricMat, matT[2] * quadricMat);
    vec4 eqCoefs =
        vec4(dot(matD[0], matT[2]), dot(matD[1], matT[2]), dot(matD[0], matT[0]), dot(matD[1], matT[1])) / dot(matD[2], matT[2]);
    vec4 outPosition = vec4(eqCoefs.x, eqCoefs.y, 0.0, 1.0);
    vec2 AABB = sqrt(eqCoefs.xy * eqCoefs.xy - eqCoefs.zw);
    AABB *= halfScreenSize * 2.0f;
    px_pos = outPosition.xy * halfScreenSize + halfScreenSize;
    pointSize = max(AABB.x, AABB.y);
}

void particle_render(
    daxa_RWBufferPtr(PackedParticleVertex) cube_rendered_particle_verts,
    daxa_RWBufferPtr(PackedParticleVertex) shadow_cube_rendered_particle_verts,
    daxa_RWBufferPtr(PackedParticleVertex) splat_rendered_particle_verts,
    daxa_RWBufferPtr(VoxelParticlesState) particles_state,
    daxa_BufferPtr(GpuInput) gpu_input, ParticleVertex vert, PackedParticleVertex packed_vertex, bool should_shadow) {
    const float voxel_radius = VOXEL_SIZE * 0.5;
    vec3 center_ws = vert.pos;
    mat4 world_to_sample = deref(gpu_input).player.cam.view_to_sample * deref(gpu_input).player.cam.world_to_view;
    vec2 half_screen_size = vec2(deref(gpu_input).frame_dim) * 0.5;
    float ps_size = 0.0;
    vec2 px_pos = vec2(0, 0);
    vec4 cs_pos = world_to_sample * vec4(center_ws, 1);
    particle_point_pos_and_size(center_ws, voxel_radius, world_to_sample, half_screen_size, px_pos, ps_size);
    if (any(lessThan(px_pos.xy + ps_size * 0.5, vec2(0))) ||
        any(greaterThan(px_pos.xy - ps_size * 0.5, vec2(half_screen_size * 2.0))) ||
        cs_pos.z / cs_pos.w < 0.0) {
        return;
    }

    const float splat_size_threshold = 5.0;
    const bool should_splat = ps_size < splat_size_threshold;

    if (ps_size <= 0.25) {
        return;
    }

#if defined(GRASS)
#define PARTICLE_RENDER_PARAMS deref(particles_state).grass
#elif defined(FLOWER)
#define PARTICLE_RENDER_PARAMS deref(particles_state).flower
#elif defined(SIM_PARTICLE)
#define PARTICLE_RENDER_PARAMS deref(particles_state).sim_particle
#elif defined(TREE_PARTICLE)
#define PARTICLE_RENDER_PARAMS deref(particles_state).tree_particle
#endif
    if (should_splat) {
        // TODO: Stochastic pruning?
        uint my_render_index = atomicAdd(PARTICLE_RENDER_PARAMS.splat_draw_params.vertex_count, 1);
        deref(advance(splat_rendered_particle_verts, my_render_index)) = packed_vertex;
    } else {
        uint my_render_index = atomicAdd(PARTICLE_RENDER_PARAMS.cube_draw_params.instance_count, 1);
        deref(advance(cube_rendered_particle_verts, my_render_index)) = packed_vertex;
    }

    if (should_shadow) {
        uint my_shadow_render_index = atomicAdd(PARTICLE_RENDER_PARAMS.shadow_cube_draw_params.instance_count, 1);
        deref(advance(shadow_cube_rendered_particle_verts, my_shadow_render_index)) = packed_vertex;
    }
}

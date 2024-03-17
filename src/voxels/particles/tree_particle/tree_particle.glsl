#pragma once

#include <utilities/gpu/noise.glsl>
#include <g_samplers>

#include "../particle.glsl"

vec3 get_tree_particle_offset(in out TreeParticle self, uint index, uint i, float time) {
    // Voxel voxel = unpack_voxel(self.packed_voxel);
    // return voxel.normal * i * VOXEL_SIZE;

    rand_seed(index * 32 + i);

    vec3 range_center = vec3(rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1);
    vec3 range_extent = vec3(1);

    return vec3(0, 0, -fract(time * 0.1 + self.origin.x)) +
           fract(vec3(vec2(time * 0.1, 0.0), 0.5) + range_center) * range_extent - range_extent * 0.5;
}

ParticleVertex get_tree_particle_vertex(daxa_BufferPtr(GpuInput) gpu_input, daxa_BufferPtr(TreeParticle) tree_particles, PackedParticleVertex packed_vertex) {
    // unpack indices from the vertex
    uint index = (packed_vertex.data >> 0) & 0xffffff;
    uint i = (packed_vertex.data >> 24) & 0xff;

    TreeParticle self = deref(tree_particles[index]);

    vec3 offset = get_tree_particle_offset(self, index, i, deref(gpu_input).time);
    vec3 prev_offset = get_tree_particle_offset(self, index, i, deref(gpu_input).time - deref(gpu_input).delta_time);

    Voxel voxel = unpack_voxel(self.packed_voxel);
    voxel.color *= i + 1;

    ParticleVertex result;
    result.pos = self.origin + offset;
    result.prev_pos = self.origin + prev_offset;
    result.packed_voxel = pack_voxel(voxel);

    result.pos = get_particle_worldspace_origin(gpu_input, result.pos);
    result.prev_pos = get_particle_prev_worldspace_origin(gpu_input, result.prev_pos);

    return result;
}

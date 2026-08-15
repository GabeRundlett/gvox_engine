#pragma once

#include <utilities/gpu/noise.glsl>
#include <renderer/globals.glsl>

#include <voxels/particles/particle.glsl>

vec3 get_fire_particle_offset(in out FireParticle self, uint index, uint i, float time) {
    rand_seed(index * 32 + i);
    if (i == 0) {
        float height = fract(time * 2.0 * (rand() + 0.5)) * 0.5;
        return vec3((good_rand(self.origin) - 0.5) * height, (good_rand(self.origin + 1) - 0.5) * height, height);
    } else {
        float height = fract(time * 2.0 * (rand() + 0.5)) * 0.5 + 1.0;
        return vec3((good_rand(self.origin) - 0.5) / height * 0.1, (good_rand(self.origin + 1) - 0.5) / height * 0.1, height * 1.5 - 1.0);
    }
}

ParticleVertex get_fire_particle_vertex(daxa_BufferPtr(GpuInput) gpu_input, daxa_BufferPtr(FireParticle) fire_particles, PackedParticleVertex packed_vertex) {
    // unpack indices from the vertex
    uint index = (packed_vertex.data >> 0) & 0xffffff;
    uint i = (packed_vertex.data >> 24) & 0xff;

    FireParticle self = deref(advance(fire_particles, index));

    vec3 offset = get_fire_particle_offset(self, index, i, deref(gpu_input).time);
    vec3 prev_offset = get_fire_particle_offset(self, index, i, deref(gpu_input).time - deref(gpu_input).delta_time);

    Voxel voxel = unpack_voxel(self.packed_voxel);

    if (i == 0) {
        voxel.roughness -= length(offset);
        voxel.roughness = clamp(voxel.roughness, 0.0, 1.0);
    } else {
        voxel.material_type = 1;
        voxel.color = vec3(0.1) + (length(offset) - 0.5) * 0.2;
    }

    ParticleVertex result;
    result.pos = self.origin + offset;
    result.prev_pos = self.origin + prev_offset;
    result.packed_voxel = pack_voxel(voxel);

    result.pos = get_particle_worldspace_origin(gpu_input, result.pos);
    result.prev_pos = get_particle_prev_worldspace_origin(gpu_input, result.prev_pos);

    return result;
}

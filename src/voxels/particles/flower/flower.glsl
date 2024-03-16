#pragma once

#include <utilities/gpu/noise.glsl>
#include <g_samplers>

#include "../particle.glsl"

vec2 flower_get_rot_offset(in out Flower self, float time) {
    // FractalNoiseConfig noise_conf = FractalNoiseConfig(
    //     /* .amplitude   = */ 1.0,
    //     /* .persistance = */ 0.5,
    //     /* .scale       = */ 0.025,
    //     /* .lacunarity  = */ 4.5,
    //     /* .octaves     = */ 4);
    // vec4 noise_val = fractal_noise(value_noise_texture, g_sampler_llr, self.origin + vec3(time * 0.5, sin(time), 0), noise_conf);
    // float rot = noise_val.x * 100.0;
    float rot = time * 5.0 + self.origin.x * (sin(time * 0.57 + self.origin.z * 1.12) * 0.125 + 0.75) + self.origin.y * (cos(time * 0.23 + self.origin.z * 0.98) * 0.125 + 0.75);
    return vec2(sin(rot), cos(rot));
}

vec3 get_flower_offset(vec2 rot_offset, float time, uint i, uint strand_index) {
    if (i <= 6) {
        // stem
        float z = float(i) * VOXEL_SIZE;
        return vec3(rot_offset * z * 0.3, z);
    } else if (i <= 6 + 27) {
        // pedals
        float z = 8 * VOXEL_SIZE;
        int xi = int(((i - 6) / 1) % 3) - 1;
        int yi = int(((i - 6) / 3) % 3) - 1;
        int zi = int(((i - 6) / 9) % 3) - 1;

        return vec3(rot_offset * z * 0.3, z) + vec3(xi, yi, zi) * VOXEL_SIZE;
    } else {
        // flakes
        rand_seed(strand_index * 32 + i);

        vec3 range_center = vec3(rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1) * VOXEL_SIZE * 6;
        vec3 range_extent = vec3(5, 5, 5);

        return fract(vec3(vec2(time * 0.1, 0.0) + rot_offset * 0.01, 0.5) + range_center) * range_extent + vec3(0, 0, 9 * VOXEL_SIZE) - range_extent * 0.5;
    }
}

void colorize_flower(in out Voxel voxel, uint i) {
    if (i <= 6) {
        // stem
        voxel.color *= i;
    } else if (i <= 6 + 27) {
        // pedals
        int xi = int(((i - 6) / 1) % 3) - 1;
        int yi = int(((i - 6) / 3) % 3) - 1;
        int zi = int(((i - 6) / 9) % 3) - 1;
        voxel.color = vec3(1, 1, 1) / 50 * float(i);
        voxel.normal = normalize(vec3(xi, yi, zi) + vec3(0, 0, 1));
    } else {
        // flakes
        voxel.color = vec3(1, 1, 1);
    }
}

ParticleVertex get_flower_vertex(daxa_BufferPtr(GpuInput) gpu_input, daxa_BufferPtr(Flower) flowers, PackedParticleVertex packed_vertex) {
    // unpack indices from the vertex
    uint strand_index = (packed_vertex.data >> 0) & 0xffffff;
    uint i = (packed_vertex.data >> 24) & 0xff;

    Flower self = deref(flowers[strand_index]);

    vec2 rot_offset = flower_get_rot_offset(self, deref(gpu_input).time);
    vec3 offset = get_flower_offset(rot_offset, deref(gpu_input).time, i, strand_index);

    vec2 prev_rot_offset = flower_get_rot_offset(self, deref(gpu_input).time - deref(gpu_input).delta_time);
    vec3 prev_offset = get_flower_offset(prev_rot_offset, deref(gpu_input).time - deref(gpu_input).delta_time, i, strand_index);

    Voxel voxel = unpack_voxel(self.packed_voxel);
    colorize_flower(voxel, i);

    ParticleVertex result;
    result.pos = self.origin + offset;
    result.prev_pos = self.origin + prev_offset;
    result.packed_voxel = pack_voxel(voxel);

    result.pos = get_particle_worldspace_origin(gpu_input, result.pos);
    result.prev_pos = get_particle_prev_worldspace_origin(gpu_input, result.prev_pos);

    return result;
}

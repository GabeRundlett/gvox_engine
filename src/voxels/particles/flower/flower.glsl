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
    float rot = time * 1.5 + self.origin.x * (sin(time * 0.157 + self.origin.z * 1.12) * 0.125 + 0.75) + self.origin.y * (cos(time * 0.063 + self.origin.z * 0.98) * 0.125 + 0.75);
    return vec2(sin(rot), cos(rot));
}

vec3 get_dandelion_offset(vec2 rot_offset, float time, uint strand_index, uint i) {
    uint height = 6;

    if (i <= height) {
        // stem
        float z = float(i) * VOXEL_SIZE;
        return vec3(rot_offset * z * 0.3, z);
    } else {
        // pedals
        float z = (1 + height) * VOXEL_SIZE;
        int xi = int((i - height - 1) % 3) - 1;
        int yi = int((i - height - 1) / 3) - 1;

        return vec3(rot_offset * z * 0.3, z) + vec3(xi, yi, 0) * VOXEL_SIZE;
    }
}
ParticleVertex process_dandelion(daxa_BufferPtr(GpuInput) gpu_input, Flower self, uint strand_index, uint i) {
    ParticleVertex result;

    // colorize
    {
        uint height = 6;
        Voxel voxel = unpack_voxel(self.packed_voxel);
        if (i <= height) {
            // stem
            voxel.color *= (i + 1) * 0.25;
            voxel.normal = vec3(0, 0, 1);
        } else {
            // pedals
            int xi = int((i - height - 1) % 3) - 1;
            int yi = int((i - height - 1) / 3) - 1;
            voxel.color = vec3(1, 0.9, 0.05);
            voxel.normal = normalize(vec3(xi, yi, 2));
        }
        result.packed_voxel = pack_voxel(voxel);
    }

    vec2 rot_offset = flower_get_rot_offset(self, deref(gpu_input).time);
    vec3 offset = get_dandelion_offset(rot_offset, deref(gpu_input).time, strand_index, i);
    result.pos = self.origin + offset;

    vec2 prev_rot_offset = flower_get_rot_offset(self, deref(gpu_input).time - deref(gpu_input).delta_time);
    vec3 prev_offset = get_dandelion_offset(prev_rot_offset, deref(gpu_input).time - deref(gpu_input).delta_time, strand_index, i);
    result.prev_pos = self.origin + prev_offset;

    return result;
}

vec3 get_dandelion_white_offset(vec2 rot_offset, float time, uint strand_index, uint i) {
    uint height = 6;

    if (i <= height) {
        // stem
        float z = float(i) * VOXEL_SIZE;
        return vec3(rot_offset * z * 0.3, z);
    } else if (i <= height + 27) {
        // pedals
        float z = (2 + height) * VOXEL_SIZE;
        int xi = int((i - height - 1) % 3) - 1;
        int yi = int(((i - height - 1) / 3) % 3) - 1;
        int zi = int((i - height - 1) / 9) - 1;

        return vec3(rot_offset * z * 0.3, z) + vec3(xi, yi, zi) * VOXEL_SIZE;
    } else {
        // flakes
        rand_seed(strand_index * 32 + i);

        vec3 range_center = vec3(rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1) * VOXEL_SIZE * height;
        vec3 range_extent = vec3(5, 5, 5);

        return fract(vec3(vec2(time * 0.1, 0.0) + rot_offset * 0.01, 0.5) + range_center) * range_extent + vec3(0, 0, 9 * VOXEL_SIZE) - range_extent * 0.5;
    }
}
ParticleVertex process_dandelion_white(daxa_BufferPtr(GpuInput) gpu_input, Flower self, uint strand_index, uint i) {
    ParticleVertex result;

    // colorize
    {
        uint height = 6;
        Voxel voxel = unpack_voxel(self.packed_voxel);
        if (i <= height) {
            // stem
            voxel.color *= (i + 1) * 0.25;
        } else if (i <= height + 27) {
            // pedals
            int xi = int((i - height - 1) % 3) - 1;
            int yi = int(((i - height - 1) / 3) % 3) - 1;
            int zi = int((i - height - 1) / 9) + 2;
            voxel.color = vec3(1, 1, 1) / 50 * float(i);
            voxel.normal = normalize(vec3(xi, yi, zi));
        } else {
            // flakes
            voxel.color = vec3(1, 1, 1);
            voxel.normal = vec3(0, 0, 1);
        }
        result.packed_voxel = pack_voxel(voxel);
    }

    vec2 rot_offset = flower_get_rot_offset(self, deref(gpu_input).time);
    vec3 offset = get_dandelion_white_offset(rot_offset, deref(gpu_input).time, strand_index, i);
    result.pos = self.origin + offset;

    vec2 prev_rot_offset = flower_get_rot_offset(self, deref(gpu_input).time - deref(gpu_input).delta_time);
    vec3 prev_offset = get_dandelion_white_offset(prev_rot_offset, deref(gpu_input).time - deref(gpu_input).delta_time, strand_index, i);
    result.prev_pos = self.origin + prev_offset;

    return result;
}

vec3 get_tulip_offset(vec2 rot_offset, float time, uint strand_index, uint i) {
    uint height = 6;

    if (i <= height) {
        // stem
        float z = float(i) * VOXEL_SIZE;
        return vec3(rot_offset * z * 0.3, z);
    } else {
        // pedals
        float z = (2 + height) * VOXEL_SIZE;
        int xi = int((i - height - 1) % 3) - 1;
        int yi = int(((i - height - 1) / 3) % 3) - 1;
        int zi = int((i - height - 1) / 9) - 1;

        return vec3(rot_offset * z * 0.3, z) + vec3(xi, yi, zi) * VOXEL_SIZE;
    }
}
ParticleVertex process_tulip(daxa_BufferPtr(GpuInput) gpu_input, Flower self, uint strand_index, uint i) {
    ParticleVertex result;

    // colorize
    {
        uint height = 6;
        Voxel voxel = unpack_voxel(self.packed_voxel);
        if (i <= height) {
            // stem
            voxel.color *= (i + 1) * 0.25;
            voxel.normal = vec3(0, 0, 1);
        } else {
            // pedals
            int xi = int((i - height - 1) % 3) - 1;
            int yi = int(((i - height - 1) / 3) % 3) - 1;
            int zi = int((i - height - 1) / 9) + 1; // bias normal up
            voxel.color = vec3(1, 0.05, 0.05);
            voxel.normal = normalize(vec3(xi, yi, zi));
        }
        result.packed_voxel = pack_voxel(voxel);
    }

    vec2 rot_offset = flower_get_rot_offset(self, deref(gpu_input).time);
    vec3 offset = get_tulip_offset(rot_offset, deref(gpu_input).time, strand_index, i);
    result.pos = self.origin + offset;

    vec2 prev_rot_offset = flower_get_rot_offset(self, deref(gpu_input).time - deref(gpu_input).delta_time);
    vec3 prev_offset = get_tulip_offset(prev_rot_offset, deref(gpu_input).time - deref(gpu_input).delta_time, strand_index, i);
    result.prev_pos = self.origin + prev_offset;

    return result;
}

vec3 get_lavender_offset(vec2 rot_offset, float time, uint strand_index, uint i) {
    uint height = 4;

    if (i <= height) {
        // stem
        float z = float(i) * VOXEL_SIZE;
        return vec3(rot_offset * z * 0.3, z);
    } else {
        // pedals
        float z = (2 + height) * VOXEL_SIZE;
        int xi = int((i - height - 1) % 3) - 1;
        int yi = int(((i - height - 1) / 3) % 3) - 1;
        int zi = int((i - height - 1) / 9) - 1;

        return vec3(rot_offset * z * 0.3, z) + vec3(xi, yi, zi) * VOXEL_SIZE;
    }
}
ParticleVertex process_lavender(daxa_BufferPtr(GpuInput) gpu_input, Flower self, uint strand_index, uint i) {
    ParticleVertex result;

    // colorize
    {
        uint height = 4;
        Voxel voxel = unpack_voxel(self.packed_voxel);
        if (i <= height) {
            // stem
            voxel.color *= (i + 1) * 0.25;
            voxel.normal = vec3(0, 0, 1);
        } else {
            // pedals
            int xi = int((i - height - 1) % 3) - 1;
            int yi = int(((i - height - 1) / 3) % 3) - 1;
            int zi = int((i - height - 1) / 9) + 2;
            voxel.color = vec3(0.2, 0.05, 1);
            voxel.normal = normalize(vec3(xi, yi, zi));
        }
        result.packed_voxel = pack_voxel(voxel);
    }

    vec2 rot_offset = flower_get_rot_offset(self, deref(gpu_input).time);
    vec3 offset = get_lavender_offset(rot_offset, deref(gpu_input).time, strand_index, i);
    result.pos = self.origin + offset;

    vec2 prev_rot_offset = flower_get_rot_offset(self, deref(gpu_input).time - deref(gpu_input).delta_time);
    vec3 prev_offset = get_lavender_offset(prev_rot_offset, deref(gpu_input).time - deref(gpu_input).delta_time, strand_index, i);
    result.prev_pos = self.origin + prev_offset;

    return result;
}

ParticleVertex get_flower_vertex(daxa_BufferPtr(GpuInput) gpu_input, daxa_BufferPtr(Flower) flowers, PackedParticleVertex packed_vertex) {
    // unpack indices from the vertex
    uint strand_index = (packed_vertex.data >> 0) & 0xffffff;
    uint i = (packed_vertex.data >> 24) & 0xff;

    Flower self = deref(advance(flowers, strand_index));

    ParticleVertex result;

    switch (self.type) {
    case FLOWER_TYPE_DANDELION: result = process_dandelion(gpu_input, self, strand_index, i); break;
    case FLOWER_TYPE_DANDELION_WHITE: result = process_dandelion_white(gpu_input, self, strand_index, i); break;
    case FLOWER_TYPE_TULIP: result = process_tulip(gpu_input, self, strand_index, i); break;
    case FLOWER_TYPE_LAVENDER: result = process_lavender(gpu_input, self, strand_index, i); break;
    }

    result.pos = get_particle_worldspace_origin(gpu_input, result.pos);
    result.prev_pos = get_particle_prev_worldspace_origin(gpu_input, result.prev_pos);

    return result;
}

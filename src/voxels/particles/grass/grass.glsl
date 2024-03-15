#pragma once

#include <utilities/gpu/noise.glsl>
#include <g_samplers>

vec2 grass_get_rot_offset(in out GrassStrand self, float time) {
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

vec3 get_grass_offset(vec2 rot_offset, float z) {
    return vec3(rot_offset * z * 0.66, z);
}

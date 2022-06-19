#pragma once

#include "utils/noise.hlsl"
#include "block_info.hlsl"

#define GROUND_LEVEL 0

float terrain_noise(float3 pos) {
    FractalNoiseConfig noise_conf = {
        /* .amplitude   = */ 1.2f,
        /* .persistance = */ 0.2f,
        /* .scale       = */ 0.004f,
        /* .lacunarity  = */ 3.5,
        /* .octaves     = */ 4,
    };
    float val = fractal_noise(pos, noise_conf) + 0.1 + (pos.y - 154 + GROUND_LEVEL) * 0.026;
    val -= smoothstep(-1, 1, (pos.y - 128 + GROUND_LEVEL) * 0.031) * 1.0;
    return val;
}

float biome_noise(float3 pos) {
    FractalNoiseConfig noise_conf = {
        /* .amplitude   = */ 1.0f,
        /* .persistance = */ 0.4f,
        /* .scale       = */ 0.001f,
        /* .lacunarity  = */ 6,
        /* .octaves     = */ 4,
    };
    return fractal_noise(float3(pos.x, 0, pos.z) + 1000, noise_conf);
}

float underworld_noise(float3 pos) {
    FractalNoiseConfig noise_conf = {
        /* .amplitude   = */ 1.0f,
        /* .persistance = */ 0.5f,
        /* .scale       = */ 0.008f,
        /* .lacunarity  = */ 2,
        /* .octaves     = */ 4,
    };
    return fractal_noise(pos, noise_conf) + abs(320 - pos.y) * 0.03 - 1.5;
}

float cave_noise(float3 pos) {
    FractalNoiseConfig noise_conf = {
        /* .amplitude   = */ 1.0f,
        /* .persistance = */ 0.4f,
        /* .scale       = */ 0.02f,
        /* .lacunarity  = */ 2,
        /* .octaves     = */ 4,
    };
    float val = fractal_noise(pos, noise_conf) + 0.1 + (254 - pos.y + GROUND_LEVEL) * 0.026;
    val -= smoothstep(-1, 1, (pos.y - 128 + GROUND_LEVEL) * 0.031) * 1.0;
    return val;
}

struct WorldgenState {
    float t_noise;
    float b_noise;
    float u_noise;
    float c_noise;

    float r;
    float r_xz;

    BlockID block_id;
    BiomeID biome_id;
};

WorldgenState get_worldgen_state(float3 pos) {
    WorldgenState result;
    result.t_noise = terrain_noise(pos);
    result.b_noise = biome_noise(pos);
    result.u_noise = underworld_noise(pos);
    result.c_noise = cave_noise(pos);
    result.r = rand(pos);
    result.r_xz = rand(float3(pos.x, 0, pos.z));
    return result;
}

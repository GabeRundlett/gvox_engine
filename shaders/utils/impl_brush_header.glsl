#pragma once

#include <utils/voxel.glsl>
#include <utils/noise.glsl>
#include <utils/sd_shapes.glsl>

struct BrushInput {
    f32vec3 p;
    f32vec3 origin;
    f32vec3 begin_p;
    Voxel prev_voxel;
};

f32 brush_noise_value(in f32vec3 voxel_p) {
    FractalNoiseConfig noise_conf = FractalNoiseConfig(
        /* .amplitude   = */ 1.0,
        /* .persistance = */ 0.5,
        /* .scale       = */ 1.0,
        /* .lacunarity  = */ 2.0,
        /* .octaves     = */ 4);
    return fractal_noise(voxel_p, noise_conf);
}

#define BRUSH_SETTINGS daxa_buffer_address_to_ref(CustomBrushSettings, push_constant.brush_settings)

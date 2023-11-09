#pragma once

#define DAXA_ENABLE_SHADER_NO_NAMESPACE 1
#define DAXA_ENABLE_IMAGE_OVERLOADS_BASIC 1
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#undef DAXA_ENABLE_SHADER_NO_NAMESPACE
#undef DAXA_ENABLE_IMAGE_OVERLOADS_BASIC

struct DensityProfileLayer {
    daxa_f32 const_term;
    daxa_f32 exp_scale;
    daxa_f32 exp_term;
    daxa_f32 layer_width;
    daxa_f32 lin_term;
};

struct SkySettings {
    daxa_f32vec3 sun_direction;

    daxa_f32 atmosphere_bottom;
    daxa_f32 atmosphere_top;

    daxa_f32vec3 mie_scattering;
    daxa_f32vec3 mie_extinction;
    daxa_f32 mie_scale_height;
    daxa_f32 mie_phase_function_g;
    DensityProfileLayer mie_density[2];

    daxa_f32vec3 rayleigh_scattering;
    daxa_f32 rayleigh_scale_height;
    DensityProfileLayer rayleigh_density[2];

    daxa_f32vec3 absorption_extinction;
    DensityProfileLayer absorption_density[2];
};

#define SKY_TRANSMITTANCE_RES u32vec2(256, 64)
#define SKY_MULTISCATTERING_RES u32vec2(32, 32)
#define SKY_SKY_RES u32vec2(192, 128)

#define MAX_SIMULATED_VOXEL_PARTICLES 0 // (1 << 14)
#define MAX_RENDERED_VOXEL_PARTICLES 0  // (1 << 14)

#define PREPASS_SCL 2
#define SHADING_SCL 2

#define ENABLE_DIFFUSE_GI false
#define ENABLE_TAA true

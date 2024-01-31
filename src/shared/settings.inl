#pragma once

#define DAXA_ENABLE_SHADER_NO_NAMESPACE 1
#define DAXA_ENABLE_IMAGE_OVERLOADS_BASIC 1
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#undef DAXA_ENABLE_SHADER_NO_NAMESPACE
#undef DAXA_ENABLE_IMAGE_OVERLOADS_BASIC

struct AutoExposureSettings {
    daxa_f32 histogram_clip_low;
    daxa_f32 histogram_clip_high;
    daxa_f32 speed;
    daxa_f32 ev_shift;
};

struct DensityProfileLayer {
    daxa_f32 const_term;
    daxa_f32 exp_scale;
    daxa_f32 exp_term;
    daxa_f32 layer_width;
    daxa_f32 lin_term;
};

struct SkySettings {
    daxa_f32vec3 sun_direction;
    daxa_f32 sun_angular_radius_cos;

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

#define SKY_TRANSMITTANCE_RES daxa_u32vec2(256, 64)
#define SKY_MULTISCATTERING_RES daxa_u32vec2(32, 32)
#define SKY_SKY_RES daxa_u32vec2(192, 192)
#define SKY_CUBE_RES 2048
#define IBL_CUBE_RES 16

#define MAX_SIMULATED_VOXEL_PARTICLES 0 // (1 << 14)
#define MAX_RENDERED_VOXEL_PARTICLES 0  // (1 << 14)

#define PREPASS_SCL 2
#define SHADING_SCL 2

#define ENABLE_TAA false

#define DEBUG_IMAGE_TYPE_DEFAULT 0
#define DEBUG_IMAGE_TYPE_DEFAULT_UINT 1
#define DEBUG_IMAGE_TYPE_GBUFFER 2
#define DEBUG_IMAGE_TYPE_SHADOW_BITMAP 3
#define DEBUG_IMAGE_TYPE_CUBEMAP 4

#define IMMEDIATE_SKY true

#define IRCACHE_GRID_CELL_DIAMETER (1.0f / 8.0f)
#define IRCACHE_CASCADE_SIZE 32
#define IRCACHE_CASCADE_COUNT 12
#define IRCACHE_USE_NORMAL_BASED_CELL_OFFSET 1

#define MAX_GRID_CELLS (IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_COUNT)
#define MAX_ENTRIES (1024 * 64)

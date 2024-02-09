#pragma once

#include <shared/voxels/voxels.inl>
#include <shared/settings.inl>

// clang-format off
#define GAME_ACTION_MOVE_FORWARD       0
#define GAME_ACTION_MOVE_LEFT          1
#define GAME_ACTION_MOVE_BACKWARD      2
#define GAME_ACTION_MOVE_RIGHT         3
#define GAME_ACTION_RELOAD             4
#define GAME_ACTION_TOGGLE_FLY         5
#define GAME_ACTION_INTERACT0          6
#define GAME_ACTION_INTERACT1          7
#define GAME_ACTION_JUMP               8
#define GAME_ACTION_CROUCH             9
#define GAME_ACTION_SPRINT             10
#define GAME_ACTION_WALK               11
#define GAME_ACTION_CYCLE_VIEW         12
#define GAME_ACTION_TOGGLE_BRUSH       13
#define GAME_ACTION_BRUSH_A            14
#define GAME_ACTION_BRUSH_B            15
#define GAME_ACTION_LAST               GAME_ACTION_BRUSH_B

#define GAME_FLAGS_PAUSED 0
#define GAME_FLAGS_NEEDS_PHYS_UPDATE 1
#define GAME_FLAG_BITS_PAUSED (daxa_u32(1) << GAME_FLAGS_PAUSED)
#define GAME_FLAG_BITS_NEEDS_PHYS_UPDATE (daxa_u32(1) << GAME_FLAGS_NEEDS_PHYS_UPDATE)

#define GAME_PHYS_UPDATE_RATE 64
#define GAME_PHYS_UPDATE_DT (1.0f / GAME_PHYS_UPDATE_RATE)
// clang-format on

struct MouseInput {
    daxa_f32vec2 pos;
    daxa_f32vec2 pos_delta;
    daxa_f32vec2 scroll_delta;
};

struct IrcacheCascadeConstants {
    daxa_i32vec4 origin;
    daxa_i32vec4 voxels_scrolled_this_frame;
};

struct GpuInput {
    daxa_u32vec2 frame_dim;
    daxa_u32vec2 rounded_frame_dim;
    daxa_u32vec2 output_resolution;
    daxa_f32vec2 halton_jitter;
    daxa_f32 pre_exposure;
    daxa_f32 pre_exposure_prev;
    daxa_f32 pre_exposure_delta;
    daxa_f32 render_res_scl;
    daxa_f32 resize_factor;
    daxa_f32 fov;
    daxa_f32 sensitivity;
    daxa_u32 frame_index;
    daxa_u32 fif_index;
    daxa_u32 flags;
    daxa_f32 time;
    daxa_f32 delta_time;
    daxa_SamplerId sampler_nnc;
    daxa_SamplerId sampler_lnc;
    daxa_SamplerId sampler_llc;
    daxa_SamplerId sampler_llr;
    daxa_f32vec3 ircache_grid_center;
    IrcacheCascadeConstants ircache_cascades[IRCACHE_CASCADE_COUNT];
    SkySettings sky_settings;
    BrushSettings world_brush_settings;
    BrushSettings brush_a_settings;
    BrushSettings brush_b_settings;
    MouseInput mouse;
    daxa_u32 actions[GAME_ACTION_LAST + 1];
};
DAXA_DECL_BUFFER_PTR(GpuInput)

struct GpuOutput {
    daxa_f32vec3 player_pos;
    daxa_f32vec3 player_rot;
    daxa_f32vec3 player_unit_offset;

    VoxelWorldOutput voxel_world;
};
DAXA_DECL_BUFFER_PTR_ALIGN(GpuOutput, 8)

#pragma once

#include <voxels/voxels.inl>
#include "settings.inl"

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

struct Camera {
    daxa_f32mat4x4 view_to_clip;
    daxa_f32mat4x4 clip_to_view;
    daxa_f32mat4x4 view_to_sample;
    daxa_f32mat4x4 sample_to_view;
    daxa_f32mat4x4 world_to_view;
    daxa_f32mat4x4 view_to_world;
    daxa_f32mat4x4 clip_to_prev_clip;
    daxa_f32mat4x4 prev_view_to_prev_clip;
    daxa_f32mat4x4 prev_clip_to_prev_view;
    daxa_f32mat4x4 prev_world_to_prev_view;
    daxa_f32mat4x4 prev_view_to_prev_world;
};

struct Player {
    Camera cam;
    daxa_f32vec3 pos; // Player (mod 1) position centered around 0.5 [0-1] (in meters)
    daxa_f32vec3 vel;
    daxa_f32 pitch, yaw, roll;
    daxa_f32vec3 forward, lateral;
    daxa_i32vec3 player_unit_offset;
    daxa_i32vec3 prev_unit_offset;
    daxa_f32 max_speed;
    daxa_u32 flags;
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
    daxa_u32 frame_index;
    daxa_u32 fif_index;
    daxa_u32 flags;
    daxa_f32 time;
    daxa_f32 delta_time;
    daxa_f32vec3 ircache_grid_center;
    IrcacheCascadeConstants ircache_cascades[IRCACHE_CASCADE_COUNT];
    SkySettings sky_settings;
    daxa_f32mat4x4 ws_to_shadow;
    daxa_f32mat4x4 shadow_to_ws;
    BrushSettings world_brush_settings;
    BrushSettings brush_a_settings;
    BrushSettings brush_b_settings;
    Player player;
    MouseInput mouse;
    daxa_u32 actions[GAME_ACTION_LAST + 1];
};
DAXA_DECL_BUFFER_PTR(GpuInput)

struct GpuOutput {
    VoxelWorldOutput voxel_world;
};
DAXA_DECL_BUFFER_PTR(GpuOutput)

struct IndirectDrawParams {
    daxa_u32 vertex_count;
    daxa_u32 instance_count;
    daxa_u32 first_vertex;
    daxa_u32 first_instance;
};

struct IndirectDrawIndexedParams {
    daxa_u32 index_count;
    daxa_u32 instance_count;
    daxa_u32 first_index;
    daxa_u32 vertex_offset;
    daxa_u32 first_instance;
};

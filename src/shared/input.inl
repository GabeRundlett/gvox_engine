#pragma once

#include <shared/core.inl>

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
#define GAME_ACTION_BREAK              14
#define GAME_ACTION_PLACE              15
#define GAME_ACTION_LAST               GAME_ACTION_PLACE
// clang-format on

struct MouseInput {
    f32vec2 pos;
    f32vec2 pos_delta;
    f32vec2 scroll_delta;
};

struct GpuInput {
    u32vec2 frame_dim;
    f32 time;
    f32 delta_time;
    f32 resize_factor;
    MouseInput mouse;
    u32 actions[GAME_ACTION_LAST + 1];
};
DAXA_ENABLE_BUFFER_PTR(GpuInput)

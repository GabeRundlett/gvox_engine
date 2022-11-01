#pragma once

#include <daxa/daxa.inl>

// clang-format off
#define GAME_MOUSE_BUTTON_1         0
#define GAME_MOUSE_BUTTON_2         1
#define GAME_MOUSE_BUTTON_3         2
#define GAME_MOUSE_BUTTON_4         3
#define GAME_MOUSE_BUTTON_5         4
#define GAME_MOUSE_BUTTON_LAST      GAME_MOUSE_BUTTON_5
#define GAME_MOUSE_BUTTON_LEFT      GAME_MOUSE_BUTTON_1
#define GAME_MOUSE_BUTTON_RIGHT     GAME_MOUSE_BUTTON_2
#define GAME_MOUSE_BUTTON_MIDDLE    GAME_MOUSE_BUTTON_3

#define GAME_KEY_MOVE_FORWARD       0
#define GAME_KEY_MOVE_LEFT          1
#define GAME_KEY_MOVE_BACKWARD      2
#define GAME_KEY_MOVE_RIGHT         3
#define GAME_KEY_RELOAD             4
#define GAME_KEY_TOGGLE_FLY         5
#define GAME_KEY_INTERACT1          6
#define GAME_KEY_INTERACT0          7
#define GAME_KEY_JUMP               8
#define GAME_KEY_CROUCH             9
#define GAME_KEY_SPRINT             10
#define GAME_KEY_WALK               11
#define GAME_KEY_CYCLE_VIEW         12
#define GAME_KEY_TOGGLE_BRUSH       13
#define GAME_KEY_LAST               GAME_KEY_TOGGLE_BRUSH
// clang-format on

struct Settings {
    u32 flags;
    f32 fov;
    f32 edit_rate;
    f32 jitter_scl;
    f32 frame_blending;
    f32 sensitivity;
    f32 daylight_cycle_time;

    f32vec3 brush_color;

    f32vec3 gen_origin;
    f32 gen_amplitude;
    f32 gen_persistance;
    f32 gen_scale;
    f32 gen_lacunarity;
    i32 gen_octaves;
};

struct MouseInput {
    f32vec2 pos;
    f32vec2 pos_delta;
    f32vec2 scroll_delta;
    u32 buttons[GAME_MOUSE_BUTTON_LAST + 1];
};

struct KeyboardInput {
    u32 keys[GAME_KEY_LAST + 1];
};

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

#define GAME_KEY_W                  0
#define GAME_KEY_A                  1
#define GAME_KEY_S                  2
#define GAME_KEY_D                  3
#define GAME_KEY_R                  4
#define GAME_KEY_F                  5
#define GAME_KEY_Q                  6
#define GAME_KEY_E                  7
#define GAME_KEY_SPACE              (GAME_KEY_E + 1)
#define GAME_KEY_LEFT_CONTROL       (GAME_KEY_E + 2)
#define GAME_KEY_LEFT_SHIFT         (GAME_KEY_E + 3)
#define GAME_KEY_LEFT_ALT           (GAME_KEY_E + 4)
#define GAME_KEY_F5                 (GAME_KEY_E + 5)
#define GAME_KEY_LAST               GAME_KEY_F5
// clang-format on

struct Settings {
    u32 flags;
    f32 fov;
    f32 edit_rate;
    f32 jitter_scl;

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

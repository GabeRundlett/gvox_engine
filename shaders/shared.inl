#pragma once

#ifdef __cplusplus
#include <daxa/daxa.hpp>

#include <daxa/utils/math_operators.hpp>
using namespace daxa::math_operators;

#else
#include "daxa/daxa.hlsl"
#endif

#define CHUNK_SIZE 64
#define CHUNK_COUNT_X 8
#define CHUNK_COUNT_Y 8
#define CHUNK_COUNT_Z 4
#define CHUNK_N (CHUNK_COUNT_X * CHUNK_COUNT_Y * CHUNK_COUNT_Z)
#define BLOCK_NX (CHUNK_COUNT_X * CHUNK_SIZE)
#define BLOCK_NY (CHUNK_COUNT_Y * CHUNK_SIZE)
#define BLOCK_NZ (CHUNK_COUNT_Z * CHUNK_SIZE)
#define BLOCK_N (CHUNK_N * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE)
#define VOXEL_SCL 8
#define CHUNK_SCL (CHUNK_SIZE / VOXEL_SCL)

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
#define GAME_KEY_SPACE              6
#define GAME_KEY_LEFT_CONTROL       7
#define GAME_KEY_LEFT_SHIFT         8
#define GAME_KEY_F5                 9
#define GAME_KEY_LAST               GAME_KEY_F5
// clang-format on

struct MouseInput {
    daxa::f32vec2 pos;
    daxa::f32vec2 pos_delta;
    daxa::f32vec2 scroll_delta;
    daxa::u32 buttons[GAME_MOUSE_BUTTON_LAST + 1];
};

struct KeyboardInput {
    daxa::u32 keys[GAME_KEY_LAST + 1];
};

struct GpuInput {
    daxa::u32vec2 render_size;
    daxa::f32vec2 jitter;
    daxa::f32 time, delta_time, fov;
    daxa::f32vec3 block_color;
    daxa::u32 flags;
    MouseInput mouse;
    KeyboardInput keyboard;
};
DAXA_DEFINE_GET_STRUCTURED_BUFFER(GpuInput);

struct GpuOutput {
    daxa::f32vec3 player_pos;
    daxa::f32vec3 player_rot;
};
DAXA_DEFINE_GET_STRUCTURED_BUFFER(GpuOutput);

struct StartupPush {
    daxa::BufferId globals_buffer_id;
};

struct PerframePush {
    daxa::BufferId globals_buffer_id;
    daxa::BufferId input_buffer_id;
    daxa::BufferId output_buffer_id;
};

struct ChunkgenPush {
    daxa::BufferId globals_buffer_id;
    daxa::BufferId input_buffer_id;
};

struct SubchunkPush {
    daxa::BufferId globals_buffer_id;
};

struct DepthPrepassPush {
    daxa::BufferId globals_buffer_id;
    daxa::BufferId input_buffer_id;
    daxa::ImageViewId render_depth_image_id;
    daxa::u32 scl;
};

struct DrawPush {
    daxa::BufferId globals_buffer_id;
    daxa::BufferId input_buffer_id;

    daxa::ImageViewId render_color_image_id;
    daxa::ImageViewId render_motion_image_id;
    daxa::ImageViewId render_depth_image_id;
};

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
#define GAME_KEY_SPACE              6
#define GAME_KEY_LEFT_CONTROL       7
#define GAME_KEY_LEFT_SHIFT         8
#define GAME_KEY_F5                 9
#define GAME_KEY_LAST               GAME_KEY_F5
// clang-format on

struct MouseInput {
    f32vec2 pos;
    f32vec2 pos_delta;
    f32vec2 scroll_delta;
    u32 buttons[GAME_MOUSE_BUTTON_LAST + 1];
};

struct KeyboardInput {
    u32 keys[GAME_KEY_LAST + 1];
};

struct Camera {
    f32mat3x3 rot_mat;
    f32vec3 pos;
    f32 fov, tan_half_fov;
};

struct Player {
    Camera cam;
    f32vec3 pos, vel;
    f32vec3 rot;
};

struct Voxel {
    f32vec3 col;
    f32vec3 nrm;
    u32 mat_id;
};

struct PackedVoxel {
    u32 data;
};

struct Sphere {
    f32vec3 o;
    f32 r;
};

struct Box {
    f32vec3 bound_min, bound_max;
};

#define CHUNK_SIZE 64

struct Chunk {
    Box box;
    PackedVoxel packed_voxels[CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE];
};

#define MAX_SPHERES 10
#define MAX_BOXES 10

struct Scene {
    u32 sphere_n;
    u32 box_n;

    Sphere spheres[MAX_SPHERES];
    Box boxes[MAX_BOXES];
    Chunk chunk;
};

DAXA_DECL_BUFFER_STRUCT(GpuInput, {
    u32vec2 frame_dim;
    f32 time;
    f32 delta_time;
    f32 fov;
    MouseInput mouse;
    KeyboardInput keyboard;
});

DAXA_DECL_BUFFER_STRUCT(GpuGlobals, {
    Player player;
    Scene scene;
});

struct StartupCompPush {
    BufferRef(GpuGlobals) gpu_globals;
};

struct PerframeCompPush {
    BufferRef(GpuGlobals) gpu_globals;
    BufferRef(GpuInput) gpu_input;
};

struct ChunkgenCompPush {
    BufferRef(GpuGlobals) gpu_globals;
};

struct DrawCompPush {
    BufferRef(GpuGlobals) gpu_globals;
    BufferRef(GpuInput) gpu_input;

    ImageViewId image_id;
};

#define SCENE push_constant.gpu_globals.scene
#define CHUNK push_constant.gpu_globals.scene.chunk

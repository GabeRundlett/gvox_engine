#pragma once

#include <shared/user_input.inl>
#include <shared/player.inl>
#include <shared/scene.inl>

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
#define VOXEL_WORLD push_constant.gpu_globals.scene.voxel_world
#define VOXEL_CHUNKS push_constant.gpu_globals.scene.voxel_world.voxel_chunks

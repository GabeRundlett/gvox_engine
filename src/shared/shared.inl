#pragma once

#define DAXA_SHADER_NO_NAMESPACE

#include <shared/user_input.inl>
#include <shared/player.inl>
#include <shared/scene.inl>

#define GPU_INPUT_FLAG_INDEX_PAUSED 0
#define GPU_INPUT_FLAG_INDEX_LIMIT_EDIT_RATE 1

DAXA_DECL_BUFFER_STRUCT(GpuInput, {
    u32vec2 frame_dim;
    f32 time;
    f32 delta_time;
    Settings settings;
    MouseInput mouse;
    KeyboardInput keyboard;
});
DAXA_DECL_BUFFER_STRUCT(GpuGlobals, {
    Player player;
    Scene scene;

    IntersectionRecord pick_intersection;
    f32vec3 brush_origin;
    f32vec3 brush_offset;

    f32vec3 edit_origin;
    u32 edit_flags;
});

DAXA_DECL_BUFFER_STRUCT(GpuIndirectDispatch, {
    u32vec3 chunk_edit_dispatch;
    u32vec3 subchunk_x2x4_dispatch;
    u32vec3 subchunk_x8up_dispatch;
});

struct GVoxModelVoxel {
    f32vec3 col;
    u32 id;
};

DAXA_DECL_BUFFER_STRUCT(GpuGVoxModel, {
    u32 size_x;
    u32 size_y;
    u32 size_z;
    GVoxModelVoxel voxels[256 * 256 * 256];
});

struct StartupCompPush {
    BufferRef(GpuGlobals) gpu_globals;
    BufferRef(VoxelWorld) voxel_world;
    // BufferRef(VoxelWorld) voxel_brush;
};
struct OpticalDepthCompPush {
    daxa_ImageViewId image_id;
};
struct PerframeCompPush {
    BufferRef(GpuGlobals) gpu_globals;
    BufferRef(GpuInput) gpu_input;
    BufferRef(VoxelWorld) voxel_world;
    // BufferRef(VoxelWorld) voxel_brush;
    BufferRef(GpuIndirectDispatch) gpu_indirect_dispatch;
};
struct ChunkOptCompPush {
    BufferRef(GpuGlobals) gpu_globals;
    BufferRef(VoxelWorld) voxel_world;
    // BufferRef(VoxelWorld) voxel_brush;
};
struct ChunkEditCompPush {
    BufferRef(GpuGlobals) gpu_globals;
    BufferRef(GpuInput) gpu_input;
    BufferRef(VoxelWorld) voxel_world;
    // BufferRef(VoxelWorld) voxel_brush;
    BufferRef(GpuGVoxModel) gpu_gvox_model;
};
struct DrawCompPush {
    BufferRef(GpuGlobals) gpu_globals;
    BufferRef(GpuInput) gpu_input;
    BufferRef(VoxelWorld) voxel_world;
    // BufferRef(VoxelWorld) voxel_brush;

    daxa_ImageViewId image_id;
    daxa_ImageViewId optical_depth_image_id;
    daxa_SamplerId optical_depth_sampler_id;
};

#define GLOBALS push_constant.gpu_globals
#define SCENE push_constant.gpu_globals.scene
#define VOXEL_WORLD push_constant.voxel_world
#define INPUT push_constant.gpu_input
#define MODEL push_constant.gpu_gvox_model
#define PLAYER push_constant.gpu_globals.player
#define INDIRECT push_constant.gpu_indirect_dispatch

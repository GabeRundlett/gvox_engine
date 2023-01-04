#pragma once

#define DAXA_ENABLE_SHADER_NO_NAMESPACE 1

#include <shared/user_input.inl>
#include <shared/player.inl>
#include <shared/scene.inl>

#define GPU_INPUT_FLAG_INDEX_PAUSED 0
#define GPU_INPUT_FLAG_INDEX_LIMIT_EDIT_RATE 1
#define GPU_INPUT_FLAG_INDEX_BRUSH_PREVIEW_OVERLAY 2
#define GPU_INPUT_FLAG_INDEX_SHOW_BRUSH_BOUNDING_BOX 3
#define GPU_INPUT_FLAG_INDEX_USE_PERSISTENT_THREAD_TRACE 4

#define USE_PERSISTENT_THREAD_TRACE 1

#define RENDER_PERF_TESTING 0

struct GVoxModelVoxel {
    f32vec3 col;
    u32 id;
};

struct GpuGlobals {
    Player player;
    Scene scene;

    IntersectionRecord pick_intersection;
    f32vec3 brush_origin;
    f32vec3 brush_offset;

    f32vec3 edit_origin;
    u32 edit_flags;

    u32vec2 padded_frame_dim;
    i32 ray_count;
};

struct GpuIndirectDispatch {
    u32vec3 chunk_edit_dispatch;
    u32vec3 subchunk_x2x4_dispatch;
    u32vec3 subchunk_x8up_dispatch;

    u32vec3 brush_chunk_dispatch;
    u32vec3 brush_subchunk_x2x4_dispatch;
    u32vec3 brush_subchunk_x8up_dispatch;

    u32vec3 raytrace_dispatch;
};

struct GpuGVoxModel {
    u32 size_x;
    u32 size_y;
    u32 size_z;
    GVoxModelVoxel voxels[256 * 256 * 256];
};

DAXA_ENABLE_BUFFER_PTR(GpuInput)
DAXA_ENABLE_BUFFER_PTR(GpuGlobals)
DAXA_ENABLE_BUFFER_PTR(GpuIndirectDispatch)
DAXA_ENABLE_BUFFER_PTR(GpuGVoxModel)

struct StartupCompPush {
    daxa_RWBufferPtr(GpuGlobals) gpu_globals;
    daxa_RWBufferPtr(VoxelWorld) voxel_world;
    daxa_RWBufferPtr(VoxelBrush) voxel_brush;
};
struct OpticalDepthCompPush {
    daxa_ImageViewId image_id;
};

#if !defined(BRUSH_INPUT)
struct CustomBrushSettings {
    u32 _x;
};
#endif
DAXA_ENABLE_BUFFER_PTR(CustomBrushSettings)

struct PerframeCompPush {
    daxa_RWBufferPtr(GpuGlobals) gpu_globals;
    daxa_BufferPtr(GpuInput) gpu_input;
    daxa_BufferPtr(CustomBrushSettings) brush_settings;
    daxa_RWBufferPtr(VoxelWorld) voxel_world;
    daxa_RWBufferPtr(VoxelBrush) voxel_brush;
    daxa_RWBufferPtr(GpuIndirectDispatch) gpu_indirect_dispatch;
};
struct ChunkOptCompPush {
    daxa_RWBufferPtr(GpuGlobals) gpu_globals;
    daxa_RWBufferPtr(VoxelWorld) voxel_world;
    daxa_RWBufferPtr(VoxelBrush) voxel_brush;
};
struct ChunkEditCompPush {
    daxa_RWBufferPtr(GpuGlobals) gpu_globals;
    daxa_BufferPtr(GpuInput) gpu_input;
    daxa_BufferPtr(CustomBrushSettings) brush_settings;
    daxa_RWBufferPtr(VoxelWorld) voxel_world;
    daxa_RWBufferPtr(VoxelBrush) voxel_brush;
    daxa_RWBufferPtr(GpuGVoxModel) gpu_gvox_model;
};
struct DrawCompPush {
    daxa_RWBufferPtr(GpuGlobals) gpu_globals;
    daxa_BufferPtr(GpuInput) gpu_input;
    daxa_RWBufferPtr(VoxelWorld) voxel_world;
    daxa_RWBufferPtr(VoxelBrush) voxel_brush;

    daxa_ImageViewId raytrace_output_image_id;
    daxa_ImageViewId image_id;
    daxa_ImageViewId optical_depth_image_id;
    daxa_SamplerId optical_depth_sampler_id;
};
struct RaytraceCompPush {
    daxa_RWBufferPtr(GpuGlobals) gpu_globals;
    daxa_BufferPtr(GpuInput) gpu_input;
    daxa_RWBufferPtr(VoxelWorld) voxel_world;
    daxa_RWBufferPtr(VoxelBrush) voxel_brush;

    daxa_ImageViewId raytrace_output_image_id;
};

#define GLOBALS deref(daxa_push_constant.gpu_globals)
#define SCENE deref(daxa_push_constant.gpu_globals).scene
#define VOXEL_WORLD deref(daxa_push_constant.voxel_world)
#define VOXEL_BRUSH deref(daxa_push_constant.voxel_brush)
#define INPUT deref(daxa_push_constant.gpu_input)
#define MODEL deref(daxa_push_constant.gpu_gvox_model)
#define PLAYER deref(daxa_push_constant.gpu_globals).player
#define INDIRECT deref(daxa_push_constant.gpu_indirect_dispatch)
#define RT_IMAGE daxa_push_constant.raytrace_output_image_id

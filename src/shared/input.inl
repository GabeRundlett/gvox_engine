#pragma once

#include <shared/voxels.inl>

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
// clang-format on

struct MouseInput {
    f32vec2 pos;
    f32vec2 pos_delta;
    f32vec2 scroll_delta;
};

struct GpuInput {
    f32 fov;
    f32 sensitivity;
    u32 log2_chunks_per_axis;
    u32 frame_index;
    u32 fif_index;
    u32vec2 frame_dim;
    u32vec2 rounded_frame_dim;
    f32vec2 halton_jitter;
    f32 time;
    f32 delta_time;
    f32 render_res_scl;
    f32 resize_factor;
    MouseInput mouse;
    u32 actions[GAME_ACTION_LAST + 1];
};
DAXA_DECL_BUFFER_PTR(GpuInput)

struct GpuOutput {
    u64 job_counters_packed;
    f32vec3 player_pos;
    f32vec3 player_rot;
    f32vec3 chunk_offset;
    u32 total_jobs_ran;

    VoxelMallocPageAllocatorGpuOutput voxel_malloc_output;
    VoxelLeafChunkAllocatorGpuOutput voxel_leaf_chunk_output;
    VoxelParentChunkAllocatorGpuOutput voxel_parent_chunk_output;
};
DAXA_DECL_BUFFER_PTR_ALIGN(GpuOutput, 8)

#pragma once

#include <shared/core.inl>

struct Camera {
    f32mat3x3 rot_mat, prev_rot_mat;
    f32vec3 pos, prev_pos;
    f32 tan_half_fov;
    f32 prev_tan_half_fov;
    f32 focal_dist;
};

struct Player {
    Camera cam;
    f32vec3 pos, vel;
    f32vec3 rot;
    f32vec3 forward, lateral;
    f32 max_speed;
};

struct VoxelChunkUpdateInfo {
    u32vec3 i;
    f32 score;
};

struct VoxelWorld {
    VoxelChunkUpdateInfo chunk_update_infos[MAX_CHUNK_UPDATES_PER_FRAME];
    u32 chunk_update_n;
};

#define CHUNK_WORK_FLAG_IS_READY_BIT (0x01)
#define CHUNK_WORK_FLAG_IS_ACTIVE_BIT (0x02)

struct ChunkNodeWorkItem {
    u32 flags;
    u32 i;
};

struct ChunkThreadPoolState {
    u64 job_counters_packed;
    u64 job_counters_packed2;
    ChunkNodeWorkItem chunk_node_work_items[MAX_NODE_THREADS];
    u32 available_threads_queue[MAX_NODE_THREADS];
    u32 total_jobs_ran;
    u32 total_jobs_ran2;
};

struct GpuIndirectDispatch {
    u32vec3 chunk_edit_dispatch;
    u32vec3 subchunk_x2x4_dispatch;
    u32vec3 subchunk_x8up_dispatch;
    u32vec3 chunk_hierarchy_dispatch;
};

struct BrushState {
    f32vec3 pos;
    f32vec3 prev_pos;
    f32vec3 initial_ray;
    u32 is_editing;
};

struct GpuGlobals {
    Player player;
    BrushState brush_state;
    VoxelWorld voxel_world;
    GpuIndirectDispatch indirect_dispatch;
    ChunkThreadPoolState chunk_thread_pool_state;
    u32 padding[10];
};
DAXA_ENABLE_BUFFER_PTR(GpuGlobals)

struct GpuSettings {
    f32 fov;
    f32 sensitivity;

    u32 log2_chunks_per_axis;
};
DAXA_ENABLE_BUFFER_PTR(GpuSettings)

#pragma once

#include <shared/core.inl>

struct IndirectDrawParams {
    u32 vertex_count;
    u32 instance_count;
    u32 first_vertex;
    u32 first_instance;
};

struct CameraLensMatrices {
    f32mat4x4 view_to_clip;
    f32mat4x4 clip_to_view;
};
struct CameraBodyMatrices {
    f32mat4x4 world_to_view;
    f32mat4x4 view_to_world;
};

struct Camera {
    f32mat4x4 proj_mat;
    f32mat3x3 rot_mat, prev_rot_mat;

    CameraLensMatrices lens_mats;
    CameraBodyMatrices body_mats;

    f32vec3 pos, prev_pos;
    f32 tan_half_fov;
    f32 prev_tan_half_fov;
    f32 focal_dist;
};

struct Player {
    Camera cam;
    f32vec3 pos; // Player (mod 8) position centered around 128 [124-132] (in meters)
    f32vec3 vel;
    f32vec3 rot;
    f32vec3 forward, lateral;
    i32vec3 chunk_offset;
    i32vec3 prev_chunk_offset;
    f32 max_speed;
};

struct BrushInput {
    f32vec3 pos;
    f32vec3 prev_pos;
};

struct VoxelChunkUpdateInfo {
    i32vec3 i;
    i32vec3 chunk_offset;
    u32 flags; // brush flags
    BrushInput brush_input;
};

struct VoxelWorld {
    VoxelChunkUpdateInfo chunk_update_infos[MAX_CHUNK_UPDATES_PER_FRAME];
    u32 chunk_update_n; // Number of chunks to update
};

struct ChunkWorkItem {
    i32vec3 i;
    i32vec3 chunk_offset;
    u32 brush_id;              // Brush ID
    BrushInput brush_input;    // Brush input parameters

    u32 children_finished[16]; // bitmask of completed children work items (16x32 = 512 children)
};

// Manages L0 and L1 ChunkWorkItems
// Values are reset between frames in perframe.comp.glsl in the following way:
// total_jobs_ran = l0_uncompleted + l1_uncompleted
// l0_queued = l0_uncompleted
// l1_queued = l1_uncompleted
// l0_completed = l0_uncompleted = l1_completed = l1_uncompleted = 0

struct ChunkThreadPoolState {
    u32 total_jobs_ran;            // total work items to run for the current frame
    u32 queue_index;               // Current queue (0: default, 1: destination for repacking unfinished work items)

    u32 work_items_l0_queued;      // Number of L0 work items in queue (also L0 dispatch x)
    u32 work_items_l0_dispatch_y;  // 1
    u32 work_items_l0_dispatch_z;  // 1

    u32 work_items_l1_queued;      // Number of L1 work items in queue (also L1 dispatch x)
    u32 work_items_l1_dispatch_y;  // 1
    u32 work_items_l1_dispatch_z;  // 1

    u32 work_items_l0_completed;   // Number of L0 work items completed for the current frame
    u32 work_items_l0_uncompleted; // Number of L0 work items left to do (one frame)
    ChunkWorkItem chunk_work_items_l0[2][MAX_CHUNK_WORK_ITEMS_L0]; // L0 work items list

    u32 work_items_l1_completed;   // Number of L1 work items completed (one frame)
    u32 work_items_l1_uncompleted; // Number of L1 work items left to do (one frame)
    ChunkWorkItem chunk_work_items_l1[2][MAX_CHUNK_WORK_ITEMS_L1]; // L1 work items list
};

struct GpuIndirectDispatch {
    u32vec3 chunk_edit_dispatch;
    u32vec3 subchunk_x2x4_dispatch;
    u32vec3 subchunk_x8up_dispatch;
};

struct BrushState {
    u32 initial_frame;
    f32vec3 initial_ray;
    u32 is_editing;
};

struct VoxelParticlesState {
    u32vec3 simulation_dispatch;
    u32 place_count;
    u32vec3 place_bounds_min;
    u32vec3 place_bounds_max;
    IndirectDrawParams draw_params;
};
struct SimulatedVoxelParticle {
    f32vec3 pos;
    f32 duration_alive;
    f32vec3 vel;
    u32 voxel_data;
    u32 flags;
};
DAXA_DECL_BUFFER_PTR(SimulatedVoxelParticle)

struct GpuGlobals {
    Player player;
    BrushInput brush_input;
    BrushState brush_state;
    VoxelWorld voxel_world;
    GpuIndirectDispatch indirect_dispatch;
    ChunkThreadPoolState chunk_thread_pool_state;
    VoxelParticlesState voxel_particles_state;
    u32 padding[10];
};
DAXA_DECL_BUFFER_PTR(GpuGlobals)

struct GpuSettings {
    f32 fov;
    f32 sensitivity;

    u32 log2_chunks_per_axis;
};
DAXA_DECL_BUFFER_PTR(GpuSettings)

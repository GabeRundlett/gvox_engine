#pragma once

#include <shared/voxels/brushes.inl>

struct IndirectDrawParams {
    u32 vertex_count;
    u32 instance_count;
    u32 first_vertex;
    u32 first_instance;
};

struct Camera {
    f32mat4x4 view_to_clip;
    f32mat4x4 clip_to_view;
    f32mat4x4 view_to_sample;
    f32mat4x4 sample_to_view;
    f32mat4x4 world_to_view;
    f32mat4x4 view_to_world;
    f32mat4x4 clip_to_prev_clip;
    f32mat4x4 prev_view_to_prev_clip;
    f32mat4x4 prev_clip_to_prev_view;
    f32mat4x4 prev_world_to_prev_view;
    f32mat4x4 prev_view_to_prev_world;
};

struct Player {
    Camera cam;
    f32vec3 pos; // Player (mod 1) position centered around 0.5 [0-1] (in meters)
    f32vec3 vel;
    f32 pitch, yaw, roll;
    f32vec3 forward, lateral;
    i32vec3 player_unit_offset;
    f32 max_speed;
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
    GpuIndirectDispatch indirect_dispatch;
    VoxelParticlesState voxel_particles_state;
};
DAXA_DECL_BUFFER_PTR(GpuGlobals)

#pragma once

#include <shared/voxels/brushes.inl>

struct IndirectDrawParams {
    daxa_u32 vertex_count;
    daxa_u32 instance_count;
    daxa_u32 first_vertex;
    daxa_u32 first_instance;
};

struct Camera {
    daxa_f32mat4x4 view_to_clip;
    daxa_f32mat4x4 clip_to_view;
    daxa_f32mat4x4 view_to_sample;
    daxa_f32mat4x4 sample_to_view;
    daxa_f32mat4x4 world_to_view;
    daxa_f32mat4x4 view_to_world;
    daxa_f32mat4x4 clip_to_prev_clip;
    daxa_f32mat4x4 prev_view_to_prev_clip;
    daxa_f32mat4x4 prev_clip_to_prev_view;
    daxa_f32mat4x4 prev_world_to_prev_view;
    daxa_f32mat4x4 prev_view_to_prev_world;
};

struct Player {
    Camera cam;
    daxa_f32vec3 pos; // Player (mod 1) position centered around 0.5 [0-1] (in meters)
    daxa_f32vec3 vel;
    daxa_f32 pitch, yaw, roll;
    daxa_f32vec3 forward, lateral;
    daxa_i32vec3 player_unit_offset;
    daxa_f32 max_speed;
};

struct GpuIndirectDispatch {
    daxa_u32vec3 chunk_edit_dispatch;
    daxa_u32vec3 subchunk_x2x4_dispatch;
    daxa_u32vec3 subchunk_x8up_dispatch;
};

struct BrushState {
    daxa_u32 initial_frame;
    daxa_f32vec3 initial_ray;
    daxa_u32 is_editing;
};

struct VoxelParticlesState {
    daxa_u32vec3 simulation_dispatch;
    daxa_u32 place_count;
    daxa_u32vec3 place_bounds_min;
    daxa_u32vec3 place_bounds_max;
    IndirectDrawParams draw_params;
};
struct SimulatedVoxelParticle {
    daxa_f32vec3 pos;
    daxa_f32 duration_alive;
    daxa_f32vec3 vel;
    daxa_u32 voxel_data;
    daxa_u32 flags;
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

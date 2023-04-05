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

struct GpuIndirectDispatch {
    u32vec3 chunk_edit_dispatch;
    u32vec3 subchunk_x2x4_dispatch;
    u32vec3 subchunk_x8up_dispatch;
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
    u32 padding[10];
};
DAXA_ENABLE_BUFFER_PTR(GpuGlobals)

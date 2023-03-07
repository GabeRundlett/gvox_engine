#pragma once

#include <shared/core.inl>

struct Camera {
    f32mat3x3 rot_mat, prev_rot_mat;
    f32vec3 pos, prev_pos;
    f32 tan_half_fov;
    f32 focal_dist;
};

struct Player {
    Camera cam;
    f32vec3 pos, vel;
    f32vec3 rot;
    f32vec3 forward, lateral;
    f32 max_speed;
};

#define MAX_CHUNK_UPDATES 10
struct VoxelWorld {
    u32vec3 chunk_update_is[MAX_CHUNK_UPDATES];
    u32 chunk_update_n;
};

struct GpuIndirectDispatch {
    u32vec3 subchunk_x2x4_dispatch;
    u32vec3 subchunk_x8up_dispatch;
};

struct GpuGlobals {
    Player player;
    VoxelWorld voxel_world;
    GpuIndirectDispatch indirect_dispatch;
};
DAXA_ENABLE_BUFFER_PTR(GpuGlobals)

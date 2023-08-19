#pragma once

#include <shared/core.inl>

struct PhysBox {
    f32 grab_dist;
    f32vec3 size;
    f32vec3 pos;
    f32vec3 vel;
    f32vec3 prev_pos;
    f32vec3 rot;
    f32vec3 rot_vel;
    f32vec3 prev_rot;
};

struct VoxelWorldGlobals {
    i32vec3 prev_offset;
    i32vec3 offset;

    PhysBox box0;
    PhysBox box1;
};
DAXA_DECL_BUFFER_PTR(VoxelWorldGlobals)

#define VOXELS_USE_BUFFERS(ptr_type, mode) \
    DAXA_TASK_USE_BUFFER(voxel_globals, ptr_type(VoxelWorldGlobals), mode)

#define VOXELS_BUFFER_USES_ASSIGN(voxel_buffers) \
    .voxel_globals = voxel_buffers.task_voxel_globals

struct VoxelWorldOutput {
    u32 _dummy;
};

struct VoxelBufferPtrs {
    daxa_BufferPtr(VoxelWorldGlobals) globals;
};
struct VoxelRWBufferPtrs {
    daxa_RWBufferPtr(VoxelWorldGlobals) globals;
};

#define VOXELS_BUFFER_PTRS VoxelBufferPtrs(daxa_BufferPtr(VoxelWorldGlobals)(voxel_globals))
#define VOXELS_RW_BUFFER_PTRS VoxelRWBufferPtrs(daxa_RWBufferPtr(VoxelWorldGlobals)(voxel_globals))

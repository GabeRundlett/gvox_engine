#pragma once

#include <shared/core.inl>

struct VoxelWorldGlobals {
    i32vec3 prev_offset;
    i32vec3 offset;
};
DAXA_DECL_BUFFER_PTR(VoxelWorldGlobals)

#define VOXELS_USE_BUFFERS(ptr_type, mode) \
    DAXA_TASK_USE_BUFFER(voxel_globals, ptr_type(u32), mode)

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

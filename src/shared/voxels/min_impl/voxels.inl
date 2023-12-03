#pragma once

#include <shared/core.inl>

struct VoxelWorldGlobals {
    daxa_i32vec3 prev_offset;
    daxa_i32vec3 offset;
};
DAXA_DECL_BUFFER_PTR(VoxelWorldGlobals)

#define VOXELS_USE_BUFFERS(ptr_type, mode) \
    DAXA_TH_BUFFER_PTR(mode, ptr_type(VoxelWorldGlobals), voxel_globals)

#define VOXELS_BUFFER_USES_ASSIGN(voxel_buffers) \
    .voxel_globals = voxel_buffers.task_voxel_globals

struct VoxelWorldOutput {
    daxa_u32 _dummy;
};

struct VoxelBufferPtrs {
    daxa_BufferPtr(VoxelWorldGlobals) globals;
};
struct VoxelRWBufferPtrs {
    daxa_RWBufferPtr(VoxelWorldGlobals) globals;
};

#define VOXELS_BUFFER_PTRS VoxelBufferPtrs(daxa_BufferPtr(VoxelWorldGlobals)(voxel_globals))
#define VOXELS_RW_BUFFER_PTRS VoxelRWBufferPtrs(daxa_RWBufferPtr(VoxelWorldGlobals)(voxel_globals))

#pragma once

#include <shared/core.inl>

struct VoxelWorldGlobals {
    u32 _dummy;
};

#define VOXELS_USE_BUFFERS(ptr_type, mode) \
    DAXA_TASK_USE_BUFFER(_dummy, ptr_type(u32), mode)

#define VOXELS_BUFFER_USES_ASSIGN(voxel_buffers) \
    ._dummy = voxel_buffers.task_dummy

struct VoxelWorldOutput {
    u32 _dummy;
};

struct VoxelBufferPtrs {
    daxa_BufferPtr(u32) _dummy;
};
struct VoxelRWBufferPtrs {
    daxa_RWBufferPtr(u32) _dummy;
};

#define VOXELS_BUFFER_PTRS VoxelBufferPtrs(daxa_BufferPtr(u32)(_dummy))
#define VOXELS_RW_BUFFER_PTRS VoxelRWBufferPtrs(daxa_RWBufferPtr(u32)(_dummy))

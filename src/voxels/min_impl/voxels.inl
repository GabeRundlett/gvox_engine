#pragma once

#include <core.inl>

struct VoxelWorldGlobals {
    daxa_i32vec3 prev_offset;
    daxa_i32vec3 offset;
};
DAXA_DECL_BUFFER_PTR(VoxelWorldGlobals)

#define VOXEL_BUFFER_USE_N 1

#define VOXELS_USE_BUFFERS(ptr_type, mode) \
    DAXA_TH_BUFFER_PTR(mode, ptr_type(VoxelWorldGlobals), voxel_globals)

#define VOXELS_USE_BUFFERS_PUSH_USES(ptr_type) \
    ptr_type(VoxelWorldGlobals) voxel_globals = push.uses.voxel_globals;

#define VOXELS_BUFFER_USES_ASSIGN(TaskHeadName, voxel_buffers)                               \
    daxa::TaskViewVariant {                                                                  \
        std::pair { TaskHeadName::voxel_globals, voxel_buffers.voxel_globals.task_resource } \
    }

struct VoxelWorldOutput {
    daxa_u32 _dummy;
};

struct VoxelBufferPtrs {
    daxa_BufferPtr(VoxelWorldGlobals) globals;
};
struct VoxelRWBufferPtrs {
    daxa_RWBufferPtr(VoxelWorldGlobals) globals;
};

#define VOXELS_BUFFER_PTRS VoxelBufferPtrs(daxa_BufferPtr(VoxelWorldGlobals)(as_address(voxel_globals)))
#define VOXELS_RW_BUFFER_PTRS VoxelRWBufferPtrs(daxa_RWBufferPtr(VoxelWorldGlobals)(as_address(voxel_globals)))

#if defined(__cplusplus)

struct VoxelWorldBuffers {
    TemporalBuffer voxel_globals;
};

#endif

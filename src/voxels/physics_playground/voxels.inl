#pragma once

#include <core.inl>

#define RIGID_BODY_FLAGS_IS_STATIC_OFFSET 0
#define RIGID_BODY_FLAG_MASK_IS_STATIC (1 << RIGID_BODY_FLAGS_IS_STATIC_OFFSET)

#define RIGID_BODY_FLAGS_SHAPE_TYPE_OFFSET 1
#define RIGID_BODY_FLAG_MASK_SHAPE_TYPE (0x7 << RIGID_BODY_FLAGS_SHAPE_TYPE_OFFSET)
#define RIGID_BODY_SHAPE_TYPE_SPHERE (0 << RIGID_BODY_FLAGS_SHAPE_TYPE_OFFSET)
#define RIGID_BODY_SHAPE_TYPE_BOX (1 << RIGID_BODY_FLAGS_SHAPE_TYPE_OFFSET)

#define RIGID_BODY_MAX_N 10

struct RigidBody {
    daxa_f32vec3 pos;
    daxa_f32vec3 lin_vel;

    // TODO: store as Rotor
    daxa_f32vec3 rot;
    daxa_f32vec3 rot_vel;

    daxa_f32vec3 size;

    float density;
    float mass;
    float restitution;

    daxa_u32 flags;
};

struct VoxelWorldGlobals {
    daxa_i32vec3 prev_offset;
    daxa_i32vec3 offset;

    daxa_u32 rigid_body_n;
    RigidBody rigid_bodies[RIGID_BODY_MAX_N];
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

#define VOXELS_BUFFER_PTRS VoxelBufferPtrs(daxa_BufferPtr(VoxelWorldGlobals)(voxel_globals))
#define VOXELS_RW_BUFFER_PTRS VoxelRWBufferPtrs(daxa_RWBufferPtr(VoxelWorldGlobals)(voxel_globals))

#if defined(__cplusplus)

struct VoxelWorldBuffers {
    TemporalBuffer voxel_globals;
};

#endif

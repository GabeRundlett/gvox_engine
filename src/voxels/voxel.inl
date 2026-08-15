#pragma once

#include <daxa/utils/task_graph.inl>
#include <voxels/defs.inl>

#ifndef __cplusplus
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_scalar_block_layout : require
#endif

struct Aabb {
    daxa_f32vec3 min;
    daxa_f32vec3 max;
};

struct Voxel {
    daxa_f32vec3 albedo;
    daxa_f32vec3 normal;
    float roughness;
    daxa_u32 material_type;
};

struct PackedVoxel {
    daxa_u32 data;
};

struct VoxelShadingAttribBrick {
    PackedVoxel voxels[BRICK_SIZE * BRICK_SIZE * BRICK_SIZE];
};
DAXA_DECL_BUFFER_PTR(VoxelShadingAttribBrick)

DAXA_FWD_DECL_BUFFER_PTR(BrickPrimitive)

struct GpuVoxelObject {
    daxa_BufferPtr(VoxelShadingAttribBrick) brick_shading_attribs;
    daxa_BufferPtr(BrickPrimitive) brick_primitives;
    daxa_f32vec3 tint;
    // daxa_BufferPtr(Aabb) brick_aabbs;
};
DAXA_DECL_BUFFER_PTR_ALIGN(GpuVoxelObject, 8)

struct BrickPrimitive {
    daxa_i32vec3 offset;
    uint8_t size_x;
    uint8_t size_y;
    uint8_t size_z;
    uint8_t flags;
    // float scale;
    daxa_u32 pad[12];
    // daxa_BufferPtr(GpuVoxelObject) voxel_object;
    uint8_t bitmap[BRICK_SIZE * BRICK_SIZE * BRICK_SIZE / 8];
};
DAXA_DECL_BUFFER_PTR_ALIGN(BrickPrimitive, 8)

DAXA_DECL_BUFFER_PTR_ALIGN(daxa_BufferPtr(BrickPrimitive), 8)

struct VoxelRtBufferPtrs {
    daxa_BufferPtr(GpuVoxelObject) voxel_object_manifests;
    daxa_u64 tlas;
};

#if defined(__cplusplus)

#include <core.inl>
// #include <utilities/allocator.inl>

struct VoxelWorldBuffers {
    TemporalBuffer voxel_object_manifests;
    daxa::ExternalTaskBuffer voxel_object_bricks;
    daxa::ExternalTaskBlas voxel_object_blases;

    uint32_t tlas_instance_count = 0;
    daxa::BufferId tlas_scratch_buffer;
    daxa::BufferId tlas_buffer;
    daxa::TlasId tlas;
    daxa::ExternalTaskTlas task_tlas;
    daxa::ExternalTaskBuffer task_tlas_instances;

    // AllocatorBufferState<VoxelLeafBrickAllocator> voxel_leaf_brick_malloc;
    // AllocatorBufferState<VoxelParentBrickAllocator> voxel_parent_brick_malloc;
};

#endif

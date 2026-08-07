#pragma once

#include <daxa/utils/task_graph.inl>
#include <voxels/defs.inl>

#ifdef __cplusplus
#else
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_scalar_block_layout : require
#endif

struct Aabb {
    daxa_f32vec3 min;
    daxa_f32vec3 max;
};

struct GpuVoxel
{
    daxa_f32vec3 albedo;
    daxa_f32vec3 normal;
    float roughness;
    daxa_u32 material_type;
};

struct GpuVoxelPackedShadingAttrib {
    uint64_t data;
};

struct VoxelShadingAttribBrick {
    GpuVoxelPackedShadingAttrib voxels[BRICK_SIZE * BRICK_SIZE * BRICK_SIZE];
};
DAXA_DECL_BUFFER_PTR(VoxelShadingAttribBrick)

DAXA_FWD_DECL_BUFFER_PTR(ChunkPrimitive)

struct GpuVoxelObject {
    daxa_BufferPtr(VoxelShadingAttribBrick) brick_shading_attribs;
    daxa_BufferPtr(ChunkPrimitive) brick_primitives;
};
DAXA_DECL_BUFFER_PTR(GpuVoxelObject)

struct ChunkPrimitive {
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
DAXA_DECL_BUFFER_PTR_ALIGN(ChunkPrimitive, 8)

DAXA_DECL_BUFFER_PTR_ALIGN(daxa_BufferPtr(ChunkPrimitive), 8)


struct VoxelRtBufferPtrs {
    daxa_BufferPtr(daxa_BufferPtr(ChunkPrimitive)) chunk_primitive_pointers;
    daxa_u64 tlas;
};

#if defined(__cplusplus)

#include <utilities/allocator.inl>

struct VoxelWorldBuffers {
    TemporalBuffer brick_primitive_pointers;
    daxa::TaskBuffer voxel_object_bricks;
    daxa::TaskBlas voxel_object_blases;

    uint32_t tlas_instance_count = 0;
    daxa::BufferId tlas_scratch_buffer;
    daxa::BufferId tlas_buffer;
    daxa::TlasId tlas;
    daxa::TaskTlas task_tlas;
    daxa::TaskBuffer task_tlas_instances;

    // AllocatorBufferState<VoxelLeafChunkAllocator> voxel_leaf_chunk_malloc;
    // AllocatorBufferState<VoxelParentChunkAllocator> voxel_parent_chunk_malloc;
};

#endif


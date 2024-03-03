#pragma once

#include <application/input.inl>
#include "voxels.inl"

DAXA_DECL_TASK_HEAD_BEGIN(VoxelWorldStartupCompute, 1 + VOXEL_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
VOXELS_USE_BUFFERS(daxa_RWBufferPtr, COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_HEAD_END
struct VoxelWorldStartupComputePush {
    DAXA_TH_BLOB(VoxelWorldStartupCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(VoxelWorldPerframeCompute, 2 + VOXEL_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(GpuOutput), gpu_output)
VOXELS_USE_BUFFERS(daxa_RWBufferPtr, COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_HEAD_END
struct VoxelWorldPerframeComputePush {
    DAXA_TH_BLOB(VoxelWorldPerframeCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(PerChunkCompute, 5)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGvoxModel), gvox_model)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_DECL_TASK_HEAD_END
struct PerChunkComputePush {
    DAXA_TH_BLOB(PerChunkCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(ChunkEditCompute, 9)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGvoxModel), gvox_model)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelMallocPageAllocator), voxel_malloc_page_allocator)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(TempVoxelChunk), temp_voxel_chunks)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, test_texture)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, test_texture2)
// DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(SimulatedVoxelParticle), simulated_voxel_particles)
// DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), placed_voxel_particles)
DAXA_DECL_TASK_HEAD_END
struct ChunkEditComputePush {
    DAXA_TH_BLOB(ChunkEditCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(ChunkEditPostProcessCompute, 7)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGvoxModel), gvox_model)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelMallocPageAllocator), voxel_malloc_page_allocator)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(TempVoxelChunk), temp_voxel_chunks)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D_ARRAY, value_noise_texture)
// DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(SimulatedVoxelParticle), simulated_voxel_particles)
// DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), placed_voxel_particles)
DAXA_DECL_TASK_HEAD_END
struct ChunkEditPostProcessComputePush {
    DAXA_TH_BLOB(ChunkEditPostProcessCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(ChunkOptCompute, 4)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(TempVoxelChunk), temp_voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_DECL_TASK_HEAD_END
struct ChunkOptComputePush {
    DAXA_TH_BLOB(ChunkOptCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(ChunkAllocCompute, 5)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(TempVoxelChunk), temp_voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelMallocPageAllocator), voxel_malloc_page_allocator)
DAXA_DECL_TASK_HEAD_END
struct ChunkAllocComputePush {
    DAXA_TH_BLOB(ChunkAllocCompute, uses)
};

#if defined(__cplusplus)

#if CPU_VOXEL_GEN
#include <glm/glm.hpp>
#endif

struct VoxelWorld {
    VoxelWorldBuffers buffers;
    daxa_u32 debug_page_count{};
    daxa_u32 debug_gpu_heap_usage{};
    bool gpu_malloc_initialized = false;

#if CPU_VOXEL_GEN
    std::vector<TempVoxelChunk> temp_voxel_chunks;
    std::vector<VoxelLeafChunk> voxel_chunks;
    VoxelWorldGlobals voxel_globals;

    Voxel get_temp_voxel(glm::ivec3 world_voxel, glm::ivec3 offset_i);
    bool has_air_neighbor(glm::ivec3 world_voxel);
    glm::vec3 generate_normal_from_geometry(glm::ivec3 world_voxel);

    void startup();
    void per_frame();
    void per_chunk(glm::uvec3 gl_GlobalInvocationID);
    void edit(glm::uvec3 gl_GlobalInvocationID);
    void edit_post_process(glm::uvec3 gl_GlobalInvocationID);
    void opt_x2x4(glm::uvec3 gl_GlobalInvocationID);
    void opt_x8up(glm::uvec3 gl_WorkGroupID, glm::uvec3 gl_GlobalInvocationID);
    void alloc(glm::uvec3 gl_GlobalInvocationID);
#endif

    void record_startup(RecordContext &record_ctx);
    void begin_frame(daxa::Device &device, VoxelWorldOutput const &gpu_output);
    void record_frame(RecordContext &record_ctx, daxa::TaskBufferView task_gvox_model_buffer, daxa::TaskImageView task_value_noise_image);

    void use_buffers(RecordContext &record_ctx);
};

#endif

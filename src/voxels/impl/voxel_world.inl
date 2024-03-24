#pragma once

#include <application/input.inl>
#include <voxels/particles/voxel_particles.inl>
#include "voxels.inl"

DAXA_DECL_TASK_HEAD_BEGIN(VoxelWorldStartupCompute, 1 + VOXEL_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
VOXELS_USE_BUFFERS(daxa_RWBufferPtr, COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_HEAD_END
struct VoxelWorldStartupComputePush {
    DAXA_TH_BLOB(VoxelWorldStartupCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(VoxelWorldPerframeCompute, 3 + VOXEL_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(GpuOutput), gpu_output)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(ChunkUpdate), chunk_updates)
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

DAXA_DECL_TASK_HEAD_BEGIN(ChunkEditCompute, 9 + SIMPLE_STATIC_ALLOCATOR_BUFFER_USE_N * 3)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGvoxModel), gvox_model)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelMallocPageAllocator), voxel_malloc_page_allocator)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(TempVoxelChunk), temp_voxel_chunks)
SIMPLE_STATIC_ALLOCATOR_USE_BUFFERS(COMPUTE_SHADER_READ_WRITE, GrassStrandAllocator)
SIMPLE_STATIC_ALLOCATOR_USE_BUFFERS(COMPUTE_SHADER_READ_WRITE, FlowerAllocator)
SIMPLE_STATIC_ALLOCATOR_USE_BUFFERS(COMPUTE_SHADER_READ_WRITE, TreeParticleAllocator)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, test_texture)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, test_texture2)
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

DAXA_DECL_TASK_HEAD_BEGIN(ChunkAllocCompute, 7)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(TempVoxelChunk), temp_voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelMallocPageAllocator), voxel_malloc_page_allocator)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(ChunkUpdate), chunk_updates)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), chunk_update_heap)
DAXA_DECL_TASK_HEAD_END
struct ChunkAllocComputePush {
    DAXA_TH_BLOB(ChunkAllocCompute, uses)
};

#if defined(__cplusplus)

struct CpuPaletteChunk {
    uint32_t variant_n{};
    uint32_t *blob_ptr{};
};

struct CpuVoxelChunk {
    std::array<CpuPaletteChunk, PALETTES_PER_CHUNK> palette_chunks{};
};

struct VoxelWorld {
    VoxelWorldBuffers buffers;
    bool gpu_malloc_initialized = false;

    std::vector<CpuVoxelChunk> voxel_chunks;

    bool sample(daxa_f32vec3 pos, daxa_i32vec3 player_unit_offset);
    void init_gpu_malloc(GpuContext &gpu_context);
    void record_startup(GpuContext &gpu_context);
    void begin_frame(daxa::Device &device, GpuInput const &gpu_input, VoxelWorldOutput const &gpu_output);
    void record_frame(GpuContext &gpu_context, daxa::TaskBufferView task_gvox_model_buffer, daxa::TaskImageView task_value_noise_image, VoxelParticles &particles);
};

#endif

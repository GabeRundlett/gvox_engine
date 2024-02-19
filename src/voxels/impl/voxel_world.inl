#pragma once

#include <application/input.inl>
#include <application/globals.inl>
#include "voxels.inl"

DAXA_DECL_TASK_HEAD_BEGIN(PerChunkCompute, 6)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGvoxModel), gvox_model)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_DECL_TASK_HEAD_END
struct PerChunkComputePush {
    DAXA_TH_BLOB(PerChunkCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(ChunkEditCompute, 10)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGlobals), globals)
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

DAXA_DECL_TASK_HEAD_BEGIN(ChunkEditPostProcessCompute, 8)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGlobals), globals)
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

DAXA_DECL_TASK_HEAD_BEGIN(ChunkOptCompute, 5)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(TempVoxelChunk), temp_voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_DECL_TASK_HEAD_END
struct ChunkOptComputePush {
    DAXA_TH_BLOB(ChunkOptCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(ChunkAllocCompute, 6)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(TempVoxelChunk), temp_voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelMallocPageAllocator), voxel_malloc_page_allocator)
DAXA_DECL_TASK_HEAD_END
struct ChunkAllocComputePush {
    DAXA_TH_BLOB(ChunkAllocCompute, uses)
};

#if defined(__cplusplus)

struct VoxelWorld {
    VoxelWorldBuffers buffers;
    daxa_u32 debug_page_count{};
    daxa_u32 debug_gpu_heap_usage{};
    bool gpu_malloc_initialized = false;

    // virtual void add_ui() override {
    //     if (ImGui::TreeNode("Voxel World")) {
    //         ImGui::Text("Page count: %u pages (%.2f MB)", debug_page_count, static_cast<double>(debug_page_count) * VOXEL_MALLOC_PAGE_SIZE_BYTES / 1'000'000.0);
    //         ImGui::Text("GPU heap usage: %.2f MB", static_cast<double>(debug_gpu_heap_usage) / 1'000'000);
    //         ImGui::TreePop();
    //     }
    // }

    void init_gpu_malloc(GpuContext &gpu_context);
    void record_startup(RecordContext &record_ctx);
    void begin_frame(daxa::Device &device, VoxelWorldOutput const &gpu_output);
    void use_buffers(RecordContext &record_ctx);
    void record_frame(RecordContext &record_ctx, daxa::TaskBufferView task_gvox_model_buffer, daxa::TaskImageView task_value_noise_image);
};

#endif

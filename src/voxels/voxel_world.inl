#pragma once

#include <application/input.inl>
#include <voxels/particles/voxel_particles.inl>
#include <voxels/voxels.inl>

DAXA_DECL_TASK_HEAD_BEGIN(VoxelWorldStartupCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
VOXELS_USE_BUFFERS(daxa_RWBufferPtr, COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_HEAD_END
struct VoxelWorldStartupComputePush {
    DAXA_TH_BLOB(VoxelWorldStartupCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(VoxelWorldPerframeCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(ChunkUpdate), chunk_updates)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_BufferPtr(BlasGeom)), geometry_pointers)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_BufferPtr(VoxelBrickAttribs)), attribute_pointers)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelBlasTransform), blas_transforms)
DAXA_TH_TLAS_PTR(COMPUTE_SHADER_READ, tlas)
VOXELS_USE_BUFFERS(daxa_RWBufferPtr, COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_HEAD_END
struct VoxelWorldPerframeComputePush {
    DAXA_TH_BLOB(VoxelWorldPerframeCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(PerChunkCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_TH_IMAGE(COMPUTE_SHADER_SAMPLED, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_DECL_TASK_HEAD_END
struct PerChunkComputePush {
    DAXA_TH_BLOB(PerChunkCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(ChunkEditCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(TempVoxelChunk), temp_voxel_chunks)
SIMPLE_STATIC_ALLOCATOR_USE_BUFFERS(COMPUTE_SHADER_READ_WRITE, GrassStrandAllocator)
SIMPLE_STATIC_ALLOCATOR_USE_BUFFERS(COMPUTE_SHADER_READ_WRITE, FlowerAllocator)
SIMPLE_STATIC_ALLOCATOR_USE_BUFFERS(COMPUTE_SHADER_READ_WRITE, TreeParticleAllocator)
SIMPLE_STATIC_ALLOCATOR_USE_BUFFERS(COMPUTE_SHADER_READ_WRITE, FireParticleAllocator)
DAXA_TH_IMAGE(COMPUTE_SHADER_SAMPLED, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, test_texture)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, test_texture2)
DAXA_DECL_TASK_HEAD_END
struct ChunkEditComputePush {
    DAXA_TH_BLOB(ChunkEditCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(ChunkEditPostProcessCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(TempVoxelChunk), temp_voxel_chunks)
DAXA_TH_IMAGE(COMPUTE_SHADER_SAMPLED, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_DECL_TASK_HEAD_END
struct ChunkEditPostProcessComputePush {
    DAXA_TH_BLOB(ChunkEditPostProcessCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(ChunkOptCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(TempVoxelChunk), temp_voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_DECL_TASK_HEAD_END
struct ChunkOptComputePush {
    DAXA_TH_BLOB(ChunkOptCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(ChunkAllocCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(TempVoxelChunk), temp_voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(ChunkUpdate), chunk_updates)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), chunk_update_heap)
DAXA_DECL_TASK_HEAD_END
struct ChunkAllocComputePush {
    DAXA_TH_BLOB(ChunkAllocCompute, uses)
};

#if defined(__cplusplus)

#include <set>

struct CpuPaletteChunk {
    uint32_t has_air : 1 {};
    uint32_t has_air_nx : 1 {};
    uint32_t has_air_ny : 1 {};
    uint32_t has_air_nz : 1 {};
    uint32_t has_air_px : 1 {};
    uint32_t has_air_py : 1 {};
    uint32_t has_air_pz : 1 {};
    uint32_t variant_n{};
    uint32_t *blob_ptr{};
};

struct BlasChunk {
    daxa::BlasId blas;
    daxa::BufferId blas_buffer;
    daxa::BufferId geom_buffer;
    daxa::BufferId attr_buffer;
    daxa::BlasBuildInfo blas_build_info;
    daxa_f32vec3 prev_position;
    daxa_f32vec3 position;
    uint32_t cull_mask;
    std::vector<BlasGeom> blas_geoms;
    std::vector<VoxelBrickAttribs> attrib_bricks;
};

struct CpuVoxelChunk {
    std::array<CpuPaletteChunk, PALETTES_PER_CHUNK> palette_chunks{};
    uint32_t blas_id;
};

using PackedDirtyChunkIndex = uint64_t;

struct VoxelWorld {
    VoxelWorldBuffers buffers;
    bool rt_initialized = false;

    std::vector<CpuVoxelChunk> voxel_chunks;

    uint32_t test_entity_blas_id;

    std::vector<BlasChunk> blas_chunks;
    std::vector<daxa::BlasId> tracked_blases;
    daxa::TaskBlas task_chunk_blases;
    TemporalBuffer staging_blas_geom_pointers;
    TemporalBuffer staging_blas_attr_pointers;
    TemporalBuffer staging_blas_transforms;
    TemporalBuffer blas_instances_buffer;
    std::set<PackedDirtyChunkIndex> dirty_chunks;

    daxa::TlasBuildInfo tlas_build_info;
    daxa::AccelerationStructureBuildSizesInfo tlas_build_sizes;
    std::array<daxa::TlasInstanceInfo, 1> blas_instance_info;

    bool sample(daxa_f32vec3 pos, daxa_i32vec3 player_unit_offset);
    void record_startup(GpuContext &gpu_context);
    void begin_frame(daxa::Device &device, GpuInput const &gpu_input);
    void update_chunks(daxa::Device &device, GpuInput const &gpu_input);
    void record_frame(GpuContext &gpu_context, VoxelParticles &particles);
};

#endif

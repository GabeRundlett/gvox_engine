#pragma once

#include "voxel_malloc.inl"
#include <voxels/gvox_model.inl>
#include <voxels/brushes.inl>

#define VOXELS_ORIGINAL_IMPL
#define INVALID_CHUNK_I ivec3(0x80000000)

// 1364 daxa_u32's
// 10.65625 bytes per 8x8x8
struct TempVoxelChunkUniformity {
    daxa_u32 lod_x2[1024];
    daxa_u32 lod_x4[256];
    daxa_u32 lod_x8[64];
    daxa_u32 lod_x16[16];
    daxa_u32 lod_x32[4];
};

// 8 bytes per 8x8x8
struct PaletteHeader {
    daxa_u32 variant_n;
    VoxelMalloc_Pointer blob_ptr;
};

struct VoxelLeafChunk {
    daxa_u32 flags;
    daxa_u32 update_index;
    daxa_u32 uniformity_bits[3];
    // 8 bytes per 8x8x8
    VoxelMalloc_ChunkLocalPageSubAllocatorState sub_allocator_state;
    // 8 bytes per 8x8x8
    PaletteHeader palette_headers[PALETTES_PER_CHUNK];
};
DAXA_DECL_BUFFER_PTR(VoxelLeafChunk)

struct TempVoxelChunk {
    PackedVoxel voxels[CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE];
    TempVoxelChunkUniformity uniformity;
};
DAXA_DECL_BUFFER_PTR(TempVoxelChunk)

struct VoxelChunkUpdateInfo {
    daxa_i32vec3 i;
    daxa_i32vec3 chunk_offset;
    daxa_u32 brush_flags;
    BrushInput brush_input;
};

struct GpuIndirectDispatch {
    daxa_u32vec3 chunk_edit_dispatch;
    daxa_u32vec3 subchunk_x2x4_dispatch;
    daxa_u32vec3 subchunk_x8up_dispatch;
};

struct BrushState {
    daxa_u32 initial_frame;
    daxa_f32vec3 initial_ray;
    daxa_u32 is_editing;
};

struct VoxelWorldGlobals {
    BrushInput brush_input;
    BrushState brush_state;
    GpuIndirectDispatch indirect_dispatch;

    VoxelChunkUpdateInfo chunk_update_infos[MAX_CHUNK_UPDATES_PER_FRAME];
    daxa_u32 chunk_update_n; // Number of chunks to update
    daxa_i32vec3 prev_offset;
    daxa_i32vec3 offset;
};
DAXA_DECL_BUFFER_PTR(VoxelWorldGlobals)

#define VOXEL_BUFFER_USE_N 3

#define VOXELS_USE_BUFFERS(ptr_type, mode)                               \
    DAXA_TH_BUFFER_PTR(mode, ptr_type(VoxelWorldGlobals), voxel_globals) \
    DAXA_TH_BUFFER_PTR(mode, ptr_type(VoxelLeafChunk), voxel_chunks)     \
    DAXA_TH_BUFFER_PTR(mode, ptr_type(VoxelMallocPageAllocator), voxel_malloc_page_allocator)

#define VOXELS_USE_BUFFERS_PUSH_USES(ptr_type)                           \
    ptr_type(VoxelWorldGlobals) voxel_globals = push.uses.voxel_globals; \
    ptr_type(VoxelLeafChunk) voxel_chunks = push.uses.voxel_chunks;      \
    ptr_type(VoxelMallocPageAllocator) voxel_malloc_page_allocator = push.uses.voxel_malloc_page_allocator;

#define VOXELS_BUFFER_USES_ASSIGN(TaskHeadName, voxel_buffers)                                                    \
    daxa::TaskViewVariant{std::pair{TaskHeadName::voxel_globals, voxel_buffers.voxel_globals.task_resource}},     \
        daxa::TaskViewVariant{std::pair{TaskHeadName::voxel_chunks, voxel_buffers.voxel_chunks.task_resource}},   \
        daxa::TaskViewVariant {                                                                                   \
        std::pair { TaskHeadName::voxel_malloc_page_allocator, voxel_buffers.voxel_malloc.task_allocator_buffer } \
    }

struct VoxelWorldOutput {
    VoxelMallocPageAllocatorGpuOutput voxel_malloc_output;
};

struct VoxelBufferPtrs {
    daxa_BufferPtr(VoxelMallocPageAllocator) allocator;
    daxa_BufferPtr(VoxelLeafChunk) voxel_chunks_ptr;
    daxa_BufferPtr(VoxelWorldGlobals) globals;
};
struct VoxelRWBufferPtrs {
    daxa_RWBufferPtr(VoxelMallocPageAllocator) allocator;
    daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunks_ptr;
    daxa_RWBufferPtr(VoxelWorldGlobals) globals;
};

#define VOXELS_BUFFER_PTRS VoxelBufferPtrs(daxa_BufferPtr(VoxelMallocPageAllocator)(as_address(voxel_malloc_page_allocator)), daxa_BufferPtr(VoxelLeafChunk)(as_address(voxel_chunks)), daxa_BufferPtr(VoxelWorldGlobals)(as_address(voxel_globals)))
#define VOXELS_RW_BUFFER_PTRS VoxelRWBufferPtrs(daxa_RWBufferPtr(VoxelMallocPageAllocator)(as_address(voxel_malloc_page_allocator)), daxa_RWBufferPtr(VoxelLeafChunk)(as_address(voxel_chunks)), daxa_RWBufferPtr(VoxelWorldGlobals)(as_address(voxel_globals)))

#if defined(__cplusplus)

#include <utilities/allocator.inl>

struct VoxelWorldBuffers {
    TemporalBuffer voxel_globals;
    TemporalBuffer voxel_chunks;
    AllocatorBufferState<VoxelMallocPageAllocator> voxel_malloc;
};

#endif

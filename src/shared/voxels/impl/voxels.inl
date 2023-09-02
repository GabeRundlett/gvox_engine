#pragma once

#include <shared/voxels/impl/voxel_malloc.inl>
#include <shared/voxels/gvox_model.inl>
#include <shared/voxels/brushes.inl>

// 1364 u32's
// 10.65625 bytes per 8x8x8
struct TempVoxelChunkUniformity {
    u32 lod_x2[1024];
    u32 lod_x4[256];
    u32 lod_x8[64];
    u32 lod_x16[16];
    u32 lod_x32[4];
};

// 8 bytes per 8x8x8
struct PaletteHeader {
    u32 variant_n;
    VoxelMalloc_Pointer blob_ptr;
};

struct VoxelParentChunk {
    u32 is_uniform;
    u32 children[512];
    u32 is_ptr[16];
};
DAXA_DECL_BUFFER_PTR(VoxelParentChunk)

struct VoxelLeafChunk {
    u32 flags;
    u32 uniformity_bits[3];
    // 8 bytes per 8x8x8
    VoxelMalloc_ChunkLocalPageSubAllocatorState sub_allocator_state;
    // 8 bytes per 8x8x8
    PaletteHeader palette_headers[PALETTES_PER_CHUNK];
};
DAXA_DECL_BUFFER_PTR(VoxelLeafChunk)

// DECL_SIMPLE_ALLOCATOR(VoxelLeafChunkAllocator, VoxelLeafChunk, 1, u32, (MAX_CHUNK_WORK_ITEMS_L2))
// DECL_SIMPLE_ALLOCATOR(VoxelParentChunkAllocator, VoxelParentChunk, 1, u32, (MAX_CHUNK_WORK_ITEMS_L0 + MAX_CHUNK_WORK_ITEMS_L1))

struct TempVoxel {
    u32 col_and_id;
};

struct TempVoxelChunk {
    TempVoxel voxels[CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE];
    TempVoxelChunkUniformity uniformity;
};
DAXA_DECL_BUFFER_PTR(TempVoxelChunk)

struct VoxelChunkUpdateInfo {
    i32vec3 i;
    u32 lod_index;
    i32vec3 chunk_offset;
    u32 brush_flags;
    BrushInput brush_input;
};

struct VoxelWorldGlobals {
    VoxelChunkUpdateInfo chunk_update_infos[MAX_CHUNK_UPDATES_PER_FRAME];
    u32 chunk_update_n; // Number of chunks to update
    i32vec3 prev_offset;
    i32vec3 offset;
};
DAXA_DECL_BUFFER_PTR(VoxelWorldGlobals)

#define VOXELS_USE_BUFFERS(ptr_type, mode)                                 \
    DAXA_TASK_USE_BUFFER(voxel_globals, ptr_type(VoxelWorldGlobals), mode) \
    DAXA_TASK_USE_BUFFER(voxel_chunks, ptr_type(VoxelLeafChunk), mode)     \
    DAXA_TASK_USE_BUFFER(voxel_malloc_page_allocator, ptr_type(VoxelMallocPageAllocator), mode)

#define VOXELS_BUFFER_USES_ASSIGN(voxel_buffers)              \
    .voxel_globals = voxel_buffers.task_voxel_globals_buffer, \
    .voxel_chunks = voxel_buffers.task_voxel_chunks_buffer,   \
    .voxel_malloc_page_allocator = voxel_buffers.voxel_malloc.task_allocator_buffer

struct VoxelWorldOutput {
    VoxelMallocPageAllocatorGpuOutput voxel_malloc_output;
    // VoxelLeafChunkAllocatorGpuOutput voxel_leaf_chunk_output;
    // VoxelParentChunkAllocatorGpuOutput voxel_parent_chunk_output;
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

#define VOXELS_BUFFER_PTRS VoxelBufferPtrs(daxa_BufferPtr(VoxelMallocPageAllocator)(voxel_malloc_page_allocator), daxa_BufferPtr(VoxelLeafChunk)(voxel_chunks), daxa_BufferPtr(VoxelWorldGlobals)(voxel_globals))
#define VOXELS_RW_BUFFER_PTRS VoxelRWBufferPtrs(daxa_RWBufferPtr(VoxelMallocPageAllocator)(voxel_malloc_page_allocator), daxa_RWBufferPtr(VoxelLeafChunk)(voxel_chunks), daxa_RWBufferPtr(VoxelWorldGlobals)(voxel_globals))

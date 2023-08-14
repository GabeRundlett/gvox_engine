#pragma once

#include <shared/voxels/impl/voxel_malloc.inl>
#include <shared/voxels/gvox_model.inl>

// 1364 u32's
// 10.65625 bytes per 8x8x8
struct VoxelChunkUniformity {
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
    VoxelChunkUniformity uniformity;
    VoxelMalloc_ChunkLocalPageSubAllocatorState sub_allocator_state;
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
};
DAXA_DECL_BUFFER_PTR(TempVoxelChunk)

struct BrushInput {
    f32vec3 pos;
    f32vec3 prev_pos;
};

struct VoxelChunkUpdateInfo {
    i32vec3 i;
    i32vec3 chunk_offset;
    u32 brush_flags;
    BrushInput brush_input;
};

struct VoxelWorldGlobals {
    VoxelChunkUpdateInfo chunk_update_infos[MAX_CHUNK_UPDATES_PER_FRAME];
    u32 chunk_update_n; // Number of chunks to update
};

#define VOXELS_USE_BUFFERS(ptr_type, mode)                             \
    DAXA_TASK_USE_BUFFER(voxel_chunks, ptr_type(VoxelLeafChunk), mode) \
    DAXA_TASK_USE_BUFFER(voxel_malloc_page_allocator, ptr_type(VoxelMallocPageAllocator), mode)

#define VOXELS_BUFFER_USES_ASSIGN(voxel_buffers)            \
    .voxel_chunks = voxel_buffers.task_voxel_chunks_buffer, \
    .voxel_malloc_page_allocator = voxel_buffers.voxel_malloc.task_allocator_buffer

struct VoxelWorldOutput {
    VoxelMallocPageAllocatorGpuOutput voxel_malloc_output;
    // VoxelLeafChunkAllocatorGpuOutput voxel_leaf_chunk_output;
    // VoxelParentChunkAllocatorGpuOutput voxel_parent_chunk_output;
};

struct VoxelBufferPtrs {
    daxa_BufferPtr(VoxelMallocPageAllocator) allocator;
    daxa_BufferPtr(VoxelLeafChunk) voxel_chunks_ptr;
};
struct VoxelRWBufferPtrs {
    daxa_RWBufferPtr(VoxelMallocPageAllocator) allocator;
    daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunks_ptr;
};

#define VOXELS_BUFFER_PTRS VoxelBufferPtrs(daxa_BufferPtr(VoxelMallocPageAllocator)(voxel_malloc_page_allocator), daxa_BufferPtr(VoxelLeafChunk)(voxel_chunks))
#define VOXELS_RW_BUFFER_PTRS VoxelRWBufferPtrs(daxa_RWBufferPtr(VoxelMallocPageAllocator)(voxel_malloc_page_allocator), daxa_RWBufferPtr(VoxelLeafChunk)(voxel_chunks))

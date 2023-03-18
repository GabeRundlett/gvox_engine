#pragma once

#include <shared/core.inl>

// One u32 per voxel, and an extra u32
#define VOXEL_MALLOC_MAX_ALLOCATION_SIZE_U32S 513
#define VOXEL_MALLOC_MAX_ALLOCATION_SIZE_BYTES (VOXEL_MALLOC_MAX_ALLOCATION_SIZE_U32S)

// Minimum size allocation is 76 bytes, aka 19 u32s
// This is because a palette of size 2 has 1 bit per
// voxel, and 2 u32s. This counts to 512 bits, plus
// 2 * 32 bits for the 2 palette entries, plus 32 bits
// for the allocation meta-data.
#define VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT 19
#define VOXEL_MALLOC_BYTES_PER_PAGE_BITFIELD_BIT (VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT * 4)
#define VOXEL_MALLOC_BITS_PER_PAGE_BITFIELD_BIT (VOXEL_MALLOC_BYTES_PER_PAGE_BITFIELD_BIT * 8)
// Because of this, the max number of allocations that
// will need to be tracked by the per-page bitfield needs
// to be at least 27. Below, we assert that this is
// large enough to hold all
#define VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD 27

#define VOXEL_MALLOC_PAGE_SIZE_U32S (VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT * VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD)
#define VOXEL_MALLOC_PAGE_SIZE_BYTES (VOXEL_MALLOC_PAGE_SIZE_U32S * 4)
#define VOXEL_MALLOC_PAGE_SIZE_BITS (VOXEL_MALLOC_PAGE_SIZE_BYTES * 8)

#if VOXEL_MALLOC_MAX_ALLOCATION_SIZE_U32S > VOXEL_MALLOC_PAGE_SIZE_U32S
#error "Not enough memory in a page to hold a max allocation!"
#endif

#define VOXEL_MALLOC_MAX_ALLOCATIONS_PER_CHUNK PALETTES_PER_CHUNK

#define VOXEL_MALLOC_MAX_PAGE_ALLOCATIONS_PER_FRAME (VOXEL_MALLOC_MAX_ALLOCATIONS_PER_CHUNK * MAX_CHUNK_UPDATES_PER_FRAME)

struct VoxelMalloc_Page {
    u32 data[VOXEL_MALLOC_PAGE_SIZE_U32S];
};
// A resizable "vector" of pages
DAXA_ENABLE_BUFFER_PTR(VoxelMalloc_Page)

// bits
//  [0-4]: chunk_local_allocator_page_index
//  [5-9]: page_bits_consumed
//  [5-31]: unused
#define VoxelMalloc_AllocationMetadata daxa_u32

// bits
//  [0-4]  local_page_alloc_offset
//  [6-31] global_page_index
#define VoxelMalloc_Pointer daxa_u32

#define VoxelMalloc_PageIndex daxa_u32
DAXA_ENABLE_BUFFER_PTR(VoxelMalloc_PageIndex)

// bits (needs to be packed as we use atomics to sync access)
//  [ 0-28]: consumption bitfield
//  [29-55]: global_page_index
//  [56-63]: unused
#define VoxelMalloc_PageInfo daxa_u64

struct VoxelMalloc_ChunkLocalPageSubAllocatorState {
    u32 page_count;
    VoxelMalloc_PageInfo page_allocation_infos[VOXEL_MALLOC_MAX_ALLOCATIONS_PER_CHUNK];
};

#define USE_OLD_ALLOC 1

#if USE_OLD_ALLOC

struct GpuAllocatorState {
    u32 offset;
};
DAXA_ENABLE_BUFFER_PTR(GpuAllocatorState)

struct VoxelMalloc_GlobalAllocator {
    daxa_RWBufferPtr(GpuAllocatorState) state;
    daxa_RWBufferPtr(daxa_u32) heap;
};

#else

struct VoxelMalloc_GlobalAllocator {
    daxa_RWBufferPtr(VoxelMalloc_Page) pages;
    daxa_RWBufferPtr(VoxelMalloc_PageIndex) available_pages_stack;
    u32 available_pages_stack_size;
    daxa_RWBufferPtr(VoxelMalloc_PageIndex) released_pages_stack;
    u32 released_pages_stack_size;
    u32 page_count;
};

#endif

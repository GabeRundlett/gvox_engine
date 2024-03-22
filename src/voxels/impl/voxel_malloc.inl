#pragma once

#include <core.inl>
#include <utilities/allocator.inl>

#define CHUNK_SIZE 64 // A chunk = 64^3 voxels
#define CHUNKS_PER_AXIS 32
#define CHUNK_LOD_LEVELS 1
#define CHUNKS_DISPATCH_SIZE (CHUNKS_PER_AXIS / 8)

#define PALETTE_REGION_SIZE 8
#define PALETTE_REGION_TOTAL_SIZE (PALETTE_REGION_SIZE * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE)
#define PALETTE_MAX_COMPRESSED_VARIANT_N 367

#if PALETTE_REGION_SIZE != 8
#error Unsupported Palette Region Size
#endif

#define PALETTES_PER_CHUNK_AXIS (CHUNK_SIZE / PALETTE_REGION_SIZE)
#define PALETTES_PER_CHUNK (PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS)

#define MAX_CHUNK_UPDATES_PER_FRAME 128

#define PALETTE_ACCELERATION_STRUCTURE_SIZE_U32S 3
// Minimum size allocation is 76 bytes, aka 19 daxa_u32s
// This is because a palette of size 2 has 1 bit per
// voxel, and 2 daxa_u32s. This counts to 512 bits, plus
// 2 * 32 bits for the 2 palette entries, plus 32 bits
// for the allocation meta-data.
#define VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT (512 / 32 + 2 + 1 + PALETTE_ACCELERATION_STRUCTURE_SIZE_U32S)
#define VOXEL_MALLOC_BYTES_PER_PAGE_BITFIELD_BIT (VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT * 4)
#define VOXEL_MALLOC_BITS_PER_PAGE_BITFIELD_BIT (VOXEL_MALLOC_BYTES_PER_PAGE_BITFIELD_BIT * 8)
// Because of this, the max number of allocations that
// will need to be tracked by the per-page bitfield needs
// to be at least 27. Below, we assert that this is
// large enough to hold the maximum allocation size of
// 2052 bytes (513 uints)
#define VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD 24
// Useful variable for knowing how many bits are necessarily
// reserved for a page-local index.
#define VOXEL_MALLOC_CEIL_LOG2_MAX_ALLOCATIONS_IN_PAGE_BITFIELD 5

#define VOXEL_MALLOC_PAGE_SIZE_U32S (VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT * VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD)
#define VOXEL_MALLOC_PAGE_SIZE_BYTES (VOXEL_MALLOC_PAGE_SIZE_U32S * 4)
#define VOXEL_MALLOC_PAGE_SIZE_BITS (VOXEL_MALLOC_PAGE_SIZE_BYTES * 8)

#define VOXEL_MALLOC_MAX_ALLOCATIONS_PER_CHUNK PALETTES_PER_CHUNK

#define VOXEL_MALLOC_MAX_PAGE_ALLOCATIONS_PER_FRAME (VOXEL_MALLOC_MAX_ALLOCATIONS_PER_CHUNK * MAX_CHUNK_UPDATES_PER_FRAME)

#define VOXEL_MALLOC_LOG2_MAX_GLOBAL_PAGE_COUNT (32 - VOXEL_MALLOC_CEIL_LOG2_MAX_ALLOCATIONS_IN_PAGE_BITFIELD)

// bits
//  [ 0-8 ]: chunk_local_allocator_page_index
//  [ 9-13]: page_bits_consumed
//  [14-31]: unused
#define VoxelMalloc_AllocationMetadata daxa_u32

// bits
// N = VOXEL_MALLOC_CEIL_LOG2_MAX_ALLOCATIONS_IN_PAGE_BITFIELD
//  [0 - N-1] local_page_alloc_offset
//  [N -  31] global_page_index
#define VoxelMalloc_Pointer daxa_u32

#define VoxelMalloc_PageIndex daxa_u32

// bits (needs to be packed as we use atomics to sync access)
// N = VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD
// M = VOXEL_MALLOC_LOG2_MAX_GLOBAL_PAGE_COUNT
//  [0 - N-1]: local_consumption_bitmask
//  [N - N+M]: global_page_index
//  [ rest? ]: unused
#define VoxelMalloc_PageInfo daxa_u64

#if (VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD + VOXEL_MALLOC_LOG2_MAX_GLOBAL_PAGE_COUNT) > 64
#error There are not enough bits in a daxa_u64 to represent the desired allocation amount, as well as the global page index.
#endif

struct VoxelMalloc_ChunkLocalPageSubAllocatorState {
    VoxelMalloc_PageInfo page_allocation_infos[VOXEL_MALLOC_MAX_ALLOCATIONS_PER_CHUNK];
};

DECL_SIMPLE_ALLOCATOR(VoxelMallocPageAllocator, daxa_u32, VOXEL_MALLOC_PAGE_SIZE_U32S, VoxelMalloc_PageIndex, (VOXEL_MALLOC_MAX_PAGE_ALLOCATIONS_PER_FRAME))

#pragma once

#if 0
#include <shared/core.inl>

struct GlobalAllocatorSettings {
    u32 heap_size;
    u32 page_size;
    u32 pages_per_heap;
    u32vec3 view_range_chunk_n;
    u32vec3 sim_range_chunk_n;

    u32 page_mask_bit_coverage;
};

struct GlobalAllocator {
    u32 allocated_page_count;
    u64 pages_memory_address;
};

struct SubchunkInfo {
    u32 chunk_local_page_index;
    u32 chunk_local_page_bits;
    u64 memory_address;
};

struct ChunkInfo {
    u64 page_allocation_infos[512];
    SubchunkInfo subchunk_infos[512];
};

#define GLOBAL_ALLOCATOR deref(daxa_push_constant.global_allocator)
#define GLOBAL_ALLOCATOR_SETTINGS deref(daxa_push_constant.global_allocator_settings)

#if defined(DAXA_SHADER)

u64 allocate_page(u32 size) {
    u32 new_page_index = atomicAdd(GLOBAL_ALLOCATOR.allocated_page_count, 1);
    return GLOBAL_ALLOCATOR.pages_memory_address + new_page_index * GLOBAL_ALLOCATOR_SETTINGS.page_size;
}

void allocate_subchunk(in out SubchunkInfo subchunk_info, u32 size) {
    u32 page_bitmask_size = (size + GLOBAL_ALLOCATOR_SETTINGS.page_mask_bit_coverage - 1) / GLOBAL_ALLOCATOR_SETTINGS.page_mask_bit_coverage;
    u32 full_size_bytes = page_bitmask_size * GLOBAL_ALLOCATOR_SETTINGS.page_mask_bit_coverage;
    u32 workgroup_index = gl_LocalInvocationIndex;
    // to reduce atomic operations, accelerate with a bitfield that
    // denotes whether a page allocation is present for the subchunk
    const u64 prior_allocation_info = atomicAdd(CHUNK_INFO.page_allocation_infos[workgroup_index], 0);
    // Search for place in the `prior_allocation_info` bitfield for an allocation large enough
    bool found_suitable_pos = /* ... */;
    u32 suitable_pos_bit_index = /* ... */;
    u32 winning_thread;
    if (found_suitable_pos) {
        // atomic elect to select thread
        winning_thread = /* ... */;
    }
}

#endif
#endif

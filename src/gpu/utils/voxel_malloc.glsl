#pragma once

#include <shared/shared.inl>

// See 'VoxelMalloc_PageInfo' in shared/voxel_malloc.inl
u32 VoxelMalloc_extract_page_mask(u64 page_info_bits) {
    return u32(page_info_bits >> 0) & ((1 << 29) - 1);
}

// See 'VoxelMalloc_PageInfo' in shared/voxel_malloc.inl
u32 VoxelMalloc_extract_global_page_index(u64 page_info_bits) {
    return u32(page_info_bits >> 29) & ((1 << 27) - 1);
}

// See 'VoxelMalloc_PageInfo' in shared/voxel_malloc.inl
VoxelMalloc_PageInfo VoxelMalloc_pack_page_info(u32 page_mask, u32 global_page_index) {
    return (u64(page_mask) << 0) | (u64(global_page_index) << 29);
}

// See 'VoxelMalloc_Pointer' in shared/voxel_malloc.inl
//  local_page_alloc_offset must be less than
VoxelMalloc_Pointer VoxelMalloc_create_pointer(u32 global_page_index, u32 local_page_alloc_offset) {
    return (local_page_alloc_offset << 0) | (global_page_index << 6);
}

// Must enter with 512 thread work group with all threads active.
shared i32 VoxelMalloc_malloc_elected_thread;
shared i32 VoxelMalloc_malloc_elected_fallback_thread;
shared i32 VoxelMalloc_malloc_elected_fallback_page_index;
shared i32 VoxelMalloc_malloc_elected_fallback_page_mask;
shared bool VoxelMalloc_malloc_allocation_success;
shared bool VoxelMalloc_malloc_global_page_index;
shared bool VoxelMalloc_malloc_global_page_mask_first_used_bit;
VoxelMalloc_Pointer VoxelMalloc_malloc(VoxelMalloc_GlobalAllocator allocator, daxa_RWBufferPtr(VoxelChunk) voxel_chunk_ptr, u32 size) {
#if USE_OLD_ALLOC
    u32 result_address = atomicAdd(deref(allocator.state).offset, size + 1);
    deref(allocator.heap[result_address]) = size + 1;
    return result_address + 1;
#else
    u32 page_bitmask_size = (size + VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT - 1) / VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT;
    // u32 full_size_bytes = page_bitmask_size * GLOBAL_ALLOCATOR_SETTINGS.page_mask_bit_coverage;
    u32 page_bitmask_no_offset = (1 << page_bitmask_size) - 1;
    u32 group_local_thread_index = gl_LocalInvocationIndex;
    if (group_local_thread_index == 0) {
        VoxelMalloc_malloc_elected_thread = -1;
        VoxelMalloc_malloc_elected_fallback_thread - 1;
        VoxelMalloc_malloc_allocation_success = false;
    }

    // First try to allocate into existing pages:
    bool page_unallocated = false;
    const u32 chunk_local_allocator_page_index = group_local_thread_index;
    do {
        const u64 const_page_info_copy = atomicAdd(deref(voxel_chunk_ptr).sub_allocator_state.page_allocation_infos[chunk_local_allocator_page_index], 0);
        const u32 page_mask = VoxelMalloc_extract_page_mask(const_page_info_copy);
        const u32 global_page_index = VoxelMalloc_extract_global_page_index(const_page_info_copy);
        page_unallocated = page_mask == 0;
        bool can_allocate = false;
        u32 page_mask_first_used_bit;
        if (!page_unallocated) {
            const u32 bit_count_before = bitCount(page_mask);
            const u32 first_zero_bit_offset = findLSB(~page_mask);
            for (u32 offset = first_zero_bit_offset; offset < (29 - page_bitmask_size); ++offset) {
                const u32 bit_mask_allocated = page_mask | (page_bitmask_no_offset << offset);
                if (bitCount(bit_mask_allocated) == bit_count_before + page_bitmask_size) {
                    page_mask_first_used_bit = offset;
                    can_allocate = true;
                    break;
                }
            }
        }
        if (can_allocate) {
            atomicCompSwap(VoxelMalloc_malloc_elected_thread, -1, group_local_thread_index);
        }
        memoryBarrierShared();
        const bool no_thread_elected = VoxelMalloc_malloc_elected_thread == -1;
        if (no_thread_elected) {
            // THIS IS NON DIVERGENT!
            break;
        } else if (VoxelMalloc_malloc_elected_fallback_thread == thread_index) {
            const u64 new_page_info_blob = VoxelMalloc_pack_page_info(page_mask, page_index);
            const u64 fetched_page_info = atomicCompSwap(deref(voxel_chunk_ptr).sub_allocator_state.page_allocation_infos[chunk_local_allocator_page_index], const_page_info_copy, new_page_metadata_blob);
            VoxelMalloc_malloc_allocation_success = fetched_page_info == const_page_info_copy;
            if (VoxelMalloc_malloc_allocation_success) {
                VoxelMalloc_malloc_global_page_mask_first_used_bit = page_mask_first_used_bit;
                VoxelMalloc_malloc_global_page_index = global_page_index;
            }
        }
    } while (!VoxelMalloc_malloc_allocation_success);
    memoryBarrierShared();
    // If allocating into existing pages fails because all current pages have too little space,
    // allocate a new page and set one of the free page metadatas to contain it.
    if (!VoxelMalloc_malloc_allocation_success) {
        // First allocate the new page and calculate the metadata for it.
        if (global_page_index == 0) {
            u32 global_page_stack_index = atomicAdd(allocator.available_pages_stack_size, -1);
            VoxelMalloc_malloc_elected_fallback_page_index = deref(allocator.available_pages_stack[global_page_stack_index]);
            VoxelMalloc_malloc_global_page_mask_first_used_bit = 0;
            VoxelMalloc_malloc_elected_fallback_page_mask = page_bitmask_no_offset;
        }
        // Then find a page meta data array element that is empty.
        // When one is found, write the metadata to it.
        memoryBarrierShared();
        do {
            if (page_unallocated) {
                // Elect ONE of the threads that map to an empty page.
                i32 elected_thread = atomicCompSwap(VoxelMalloc_malloc_elected_thread, -1, group_local_thread_index);
                if (elected_thread == group_local_thread_index) {
                    // Pack metadata.
                    const u64 new_page_info_blob = VoxelMalloc_pack_page_info(VoxelMalloc_malloc_elected_fallback_page_mask, VoxelMalloc_malloc_elected_fallback_page_index);
                    // Try to write metadata to the elected page meta info.
                    const u64 fetched_page_info = atomicCompSwap(deref(voxel_chunk_ptr).sub_allocator_state.page_allocation_infos[chunk_local_allocator_page_index], u64(0), new_page_metadata_blob);
                    // We succeed, when the page meta info was 0 (meaning it was empty) in the atomic comp swap.
                    // If we get back a 0, we have successfully written the page meta info.
                    VoxelMalloc_malloc_allocation_success = fetched_page_info == u64(0);
                    // It can happen that another workgroup changes the metadata of our empty page, as everything is parallel here.
                    // If that happenes we need to update the boolean determining if we can update the meta data of the page the thread maps to.
                    page_unallocated = fetched_page_info == u64(0);
                    if (VoxelMalloc_malloc_allocation_success) {
                        // TODO(grundlett): Write chunk_local_allocator_page_index into the allocations header in here!
                        daxa_RWBufferPtr(VoxelMalloc_Page) page = allocator.pages[VoxelMalloc_malloc_elected_fallback_page_index];
                        deref(page.data[VoxelMalloc_malloc_global_page_mask_first_used_bit * VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT]) = (chunk_local_allocator_page_index << 0) | (page_bitmask_size << 5);
                    }
                }
            }
            memoryBarrierShared();
        } while (!VoxelMalloc_malloc_allocation_success);
    }

    return VoxelMalloc_create_pointer(VoxelMalloc_malloc_global_page_index, VoxelMalloc_malloc_global_page_mask_first_used_bit);
#endif
}

// Can enter with any workgroup and warp configuration.
void VoxelMalloc_free(VoxelMalloc_GlobalAllocator allocator, daxa_RWBufferPtr(VoxelChunk) voxel_chunk_ptr, VoxelMalloc_Pointer address) {
#if USE_OLD_ALLOC
    // Doesn't matter for now...

    // if (address != 0) {
    //     i32 size = i32(deref(allocator.heap[address - 1]));
    //     atomicAdd(deref(allocator.state).offset, -size);
    // }
#else
    const u32 global_page_index = address >> 5;

    // Only ONE thread should ever enter this!
    daxa_RWBufferPtr(VoxelMalloc_Page) page = allocator.pages[global_page_index];
    u32 local_page_alloc_offset = address & 0x1f;

    const u32 chunk_local_allocator_page_index = deref(page.data[local_page_alloc_offset * VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT]) & 0x1f;

    // The idea behind the synchronization here:
    // We fetch the metadata with a single atomic operation.
    // Then we modify a local copy of the metadata.
    // After we are done with our modifications we *try* to write it back.
    // To try we do an atomicCompSwap with the initialy fetched value for comparison.
    // If the values are the same this means that out changes are valid as the effect would have been as if this whole thing would have happened atomically.
    // If the values are NOT the same, another thread modified the data in a way that makes our changes INVALID.
    bool deallocate_page = false;
    do {
        const u64 const_page_info_copy = atomicAdd(deref(voxel_chunk_ptr).sub_allocator_state.page_allocation_infos[chunk_local_allocator_page_index], 0);
        const u32 page_mask = VoxelMalloc_extract_page_mask(const_page_info_copy);
        const u32 global_page_index = VoxelMalloc_extract_global_page_index(const_page_info_copy);

        const u32 allocation_bits = ...;
        const u32 page_mask_deallocated = global_page_index & ~(allocation_bits);
        deallocate_page = page_mask_deallocated == 0;

        u64 our_page_metadata_blob = 0;
        if (page_empty_after_deallocation) {
            our_page_metadata_blob = ...; // Set the page index to a value indicating that the allocation is empty.
            deallocate_page = true;
        } else {
            our_page_metadata_blob = ...;
        }

        const u64 fetched_page_metadata = atomicCompSwap(deref(voxel_chunk_ptr).sub_allocator_state.page_allocation_infos[chunk_local_allocator_page_index], const_page_info_copy, our_page_metadata_blob);
        const bool free_successfull = fetched_page_metadata == const_page_info_copy;
    } while (!free_successfull);

    if (deallocate_page) {
        const u32 free_stack_index = atomicAdd(deref(allocator).freed_pages_stage_size, 1);
        deref(allocator).available_pages_stack[free_stack_index] = global_page_index;
    }
#endif
}

daxa_RWBufferPtr(daxa_u32) voxel_malloc_address_to_u32_ptr(VoxelMalloc_GlobalAllocator allocator, VoxelMalloc_Pointer address, u32 offset) {
#if USE_OLD_ALLOC
    daxa_RWBufferPtr(daxa_u32) result;
    return result;
#else
    daxa_RWBufferPtr(VoxelMalloc_Page) page = allocator.pages[address >> 5];
    daxa_RWBufferPtr(daxa_u32) result = daxa_RWBufferPtr(daxa_u32)(page);
    return result + (address & 0x1f) * VOXEL_MALLOC_MAX_ALLOCATION_SIZE_U32S;
#endif
}

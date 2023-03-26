#pragma once

#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_debug_printf : require

#include <shared/shared.inl>

// See 'VoxelMalloc_PageInfo' in shared/voxel_malloc.inl
u32 VoxelMalloc_PageInfo_extract_local_consumption_bitmask(VoxelMalloc_PageInfo page_info_bits) {
    return u32(page_info_bits >> 0) & ((1 << VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD) - 1);
}
u32 VoxelMalloc_PageInfo_extract_global_page_index(VoxelMalloc_PageInfo page_info_bits) {
    return u32(page_info_bits >> VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD) & ((1 << 27) - 1);
}
VoxelMalloc_PageInfo VoxelMalloc_PageInfo_pack(u32 page_local_consumption_bitmask, u32 global_page_index) {
    return (u64(page_local_consumption_bitmask) << 0) | (u64(global_page_index) << VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD);
}

// See 'VoxelMalloc_AllocationMetadata' in shared/voxel_malloc.inl
u32 VoxelMalloc_AllocationMetadata_extract_chunk_local_allocator_page_index(VoxelMalloc_AllocationMetadata metadata) {
    return (metadata >> 0) & 0x1ff;
}
u32 VoxelMalloc_AllocationMetadata_extract_page_bits_consumed(VoxelMalloc_AllocationMetadata metadata) {
    return (metadata >> 9);
}
VoxelMalloc_AllocationMetadata VoxelMalloc_AllocationMetadata_pack(u32 chunk_local_allocator_page_index, u32 page_bits_consumed) {
    return (chunk_local_allocator_page_index << 0) | (page_bits_consumed << 9);
}

// See 'VoxelMalloc_Pointer' in shared/voxel_malloc.inl
u32 VoxelMalloc_Pointer_extract_local_page_alloc_offset(VoxelMalloc_Pointer ptr) {
    return (ptr >> 0) & 0x1f;
}
u32 VoxelMalloc_Pointer_extract_global_page_index(VoxelMalloc_Pointer ptr) {
    return (ptr >> 5);
}
VoxelMalloc_Pointer VoxelMalloc_Pointer_pack(u32 global_page_index, u32 local_page_alloc_offset) {
    return (local_page_alloc_offset << 0) | (global_page_index << 5);
}

// Must enter with 512 thread work group with all threads active.
shared i32 VoxelMalloc_malloc_elected_thread;
shared i32 VoxelMalloc_malloc_elected_fallback_thread;
shared i32 VoxelMalloc_malloc_elected_fallback_page_local_consumption_bitmask;
shared bool VoxelMalloc_malloc_allocation_success;
shared u32 VoxelMalloc_malloc_global_page_index;
shared u32 VoxelMalloc_malloc_page_local_consumption_bitmask_first_used_bit;
#define PAGE_ALLOC_INFOS(i) deref(voxel_chunk_ptr).sub_allocator_state.page_allocation_infos[i]
VoxelMalloc_Pointer VoxelMalloc_malloc(daxa_RWBufferPtr(VoxelMalloc_GlobalAllocator) allocator, daxa_RWBufferPtr(VoxelChunk) voxel_chunk_ptr, u32 size) {
#if USE_OLD_ALLOC
    u32 result_address = atomicAdd(deref(allocator).offset, size + 1);
    deref(deref(allocator).heap[result_address]) = size + 1;
    return result_address + 1;
#else
    u32 local_allocation_bit_n = (size + 1 + VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT - 1) / VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT;
    u32 local_allocation_bitmask_no_offset = (1 << local_allocation_bit_n) - 1;
    i32 group_local_thread_index = i32(gl_LocalInvocationIndex);
    if (group_local_thread_index == 0) {
        VoxelMalloc_malloc_elected_thread = -1;
        VoxelMalloc_malloc_elected_fallback_thread = -1;
        VoxelMalloc_malloc_allocation_success = false;
    }

    // First try to allocate into existing pages:
    bool page_unallocated = false;
    bool did_it_already = false;
    bool did_it_already1 = false;
    const u32 chunk_local_allocator_page_index = group_local_thread_index;
    while (true) {
        memoryBarrierShared();
        barrier();
        const VoxelMalloc_PageInfo const_page_info_copy = atomicAdd(PAGE_ALLOC_INFOS(chunk_local_allocator_page_index), 0);
        const u32 page_local_consumption_bitmask_before = VoxelMalloc_PageInfo_extract_local_consumption_bitmask(const_page_info_copy);
        const u32 global_page_index = VoxelMalloc_PageInfo_extract_global_page_index(const_page_info_copy);
        page_unallocated = page_local_consumption_bitmask_before == 0;
        bool can_allocate = false;
        u32 page_local_consumption_bitmask_first_used_bit;
        u32 page_local_consumption_bitmask_with_new_allocation = 0;
        if (!page_unallocated) {
            const u32 bit_count_before = bitCount(page_local_consumption_bitmask_before);
            const u32 first_zero_bit_offset = findLSB(~page_local_consumption_bitmask_before);
            for (u32 offset = first_zero_bit_offset; offset < (VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD - local_allocation_bit_n); ++offset) {
                const u32 potential_local_allocation_bitmask = page_local_consumption_bitmask_before | (local_allocation_bitmask_no_offset << offset);
                if (bitCount(potential_local_allocation_bitmask) == bit_count_before + local_allocation_bit_n) {
                    page_local_consumption_bitmask_first_used_bit = offset;
                    page_local_consumption_bitmask_with_new_allocation = potential_local_allocation_bitmask;
                    can_allocate = true;
                    break;
                }
            }
        }
        if (can_allocate) {
            i32 election_fetch = atomicCompSwap(VoxelMalloc_malloc_elected_thread, -1, group_local_thread_index);
            if (election_fetch == -1) {
                // Thread won election.
                // Try to publish the allocation into the local page.
                // Construct new page info blob.
                const u64 new_page_info = VoxelMalloc_PageInfo_pack(page_local_consumption_bitmask_with_new_allocation, global_page_index);
                // Try to finalize the allocation. The allocation succeses, when the fetched page
                // info we got in the beginning did not change up until this atomic operation.
                // If the page info changed, we need to start over as we worked on outdated information for this attempt.
                const u64 fetched_page_info = atomicCompSwap(PAGE_ALLOC_INFOS(chunk_local_allocator_page_index), const_page_info_copy, new_page_info);
                VoxelMalloc_malloc_allocation_success = (fetched_page_info == const_page_info_copy);
                if (VoxelMalloc_malloc_allocation_success) {
                    VoxelMalloc_malloc_page_local_consumption_bitmask_first_used_bit = page_local_consumption_bitmask_first_used_bit;
                    VoxelMalloc_malloc_global_page_index = global_page_index;
                    daxa_RWBufferPtr(VoxelMalloc_Page) page = deref(allocator).pages[VoxelMalloc_malloc_global_page_index];
                    deref(page).data[page_local_consumption_bitmask_first_used_bit * VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT] = VoxelMalloc_AllocationMetadata_pack(chunk_local_allocator_page_index, local_allocation_bit_n);
                }
            }
        }
        // NOTE: Potentially speed up by using 2 `VoxelMalloc_malloc_elected_thread`'s, removing
        // the need for the second set of barriers.
        barrier();
        memoryBarrierShared();
        // We need to break and allocate a new page in the fallback loop below.
        // If no thread got elected to try and allocate, its impossible to allocate into an existing palette.
        const bool allocation_impossible = VoxelMalloc_malloc_elected_thread == -1;
        const bool uniform_breaking_condition = allocation_impossible || VoxelMalloc_malloc_allocation_success;
        barrier();
        if (uniform_breaking_condition) {
            // THIS IS NON DIVERGENT!
            break;
        } else if (group_local_thread_index == 0) {
            VoxelMalloc_malloc_elected_thread = -1;
        }
    }
    // If allocating into existing pages fails because all current pages have too little space,
    // allocate a new page and set one of the free page metadatas to contain it.
    if (!VoxelMalloc_malloc_allocation_success) {
        // First allocate the new page and calculate the metadata for it.
        if (group_local_thread_index == 0) {
            // Try to get a page from the available stack.
            i32 global_page_stack_index = atomicAdd(deref(allocator).available_pages_stack_size, -1) - 1;
            // If we get an index of smaller then zero, the size was 0. This means the stack was empty.
            // In that case we need to create new pages as the available stack is empty.
            if (global_page_stack_index < 0) {
                // Create new page.
                VoxelMalloc_malloc_global_page_index = atomicAdd(deref(allocator).page_count, 1);
            } else {
                VoxelMalloc_malloc_global_page_index = i32(deref(deref(allocator).available_pages_stack[global_page_stack_index]));
            }
            VoxelMalloc_malloc_page_local_consumption_bitmask_first_used_bit = 0;
        }
        // Then find a page meta data array element that is empty.
        // When one is found, write the metadata to it.
        barrier();
        memoryBarrierShared();
        i32 loop_iter = 0;
        while (!VoxelMalloc_malloc_allocation_success) {
            barrier();
            memoryBarrierShared();

            ++loop_iter;
            if (page_unallocated) {
                // Elect ONE of the threads that map to an empty page.
                i32 election_fetch = atomicCompSwap(VoxelMalloc_malloc_elected_thread, -1, group_local_thread_index);
                if (election_fetch == -1) {
                    // Pack metadata.
                    const u64 new_page_info = VoxelMalloc_PageInfo_pack(local_allocation_bitmask_no_offset, VoxelMalloc_malloc_global_page_index);
                    // Try to write metadata to the elected page meta info.
                    const u64 fetched_page_info = atomicCompSwap(PAGE_ALLOC_INFOS(chunk_local_allocator_page_index), u64(0), new_page_info);
                    // We succeed, when the page meta info was 0 (meaning it was empty) in the atomic comp swap.
                    // If we get back a 0, we have successfully written the page meta info.
                    VoxelMalloc_malloc_allocation_success = fetched_page_info == u64(0);
                    // It can happen that another workgroup changes the metadata of our empty page, as everything is parallel here.
                    // If that happenes we need to update the boolean determining if we can update the meta data of the page the thread maps to.
                    page_unallocated = fetched_page_info == u64(0);
                    if (VoxelMalloc_malloc_allocation_success) {
                        daxa_RWBufferPtr(VoxelMalloc_Page) page = deref(allocator).pages[VoxelMalloc_malloc_global_page_index];
                        deref(page).data[0] = VoxelMalloc_AllocationMetadata_pack(chunk_local_allocator_page_index, local_allocation_bit_n);
                    }
                }
            }
            barrier();
            memoryBarrierShared();
            // Reset the thread voting value for the text allocation attempt.
            // TODO: use two election values to avoid a memory and execution barrier to reset the election value.
            if (group_local_thread_index == 0) {
                VoxelMalloc_malloc_elected_thread = -1;
            }
        }
    }

    barrier();
    memoryBarrierShared();

    VoxelMalloc_Pointer blob_ptr = VoxelMalloc_Pointer_pack(VoxelMalloc_malloc_global_page_index, VoxelMalloc_malloc_page_local_consumption_bitmask_first_used_bit);
    return blob_ptr;
#endif
}
#undef PAGE_ALLOC_INFOS

// Can enter with any workgroup and warp configuration.
#define PAGE_ALLOC_INFOS(i) deref(voxel_chunk_ptr).sub_allocator_state.page_allocation_infos[i]
void VoxelMalloc_free(daxa_RWBufferPtr(VoxelMalloc_GlobalAllocator) allocator, daxa_RWBufferPtr(VoxelChunk) voxel_chunk_ptr, VoxelMalloc_Pointer address) {
#if USE_OLD_ALLOC
    // Doesn't matter for now...

    // if (address != 0) {
    //     i32 size = i32(deref(deref(allocator).heap[address - 1]));
    //     atomicAdd(deref(allocator.state).offset, -size);
    // }
#else
    // return;

    const u32 global_page_index = VoxelMalloc_Pointer_extract_global_page_index(address);
    u32 local_page_alloc_offset = VoxelMalloc_Pointer_extract_local_page_alloc_offset(address);

    // Only ONE thread should ever enter this!
    daxa_RWBufferPtr(VoxelMalloc_Page) page = deref(allocator).pages[global_page_index];

    const VoxelMalloc_AllocationMetadata allocation_metadata = deref(page).data[local_page_alloc_offset * VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT];
    const u32 chunk_local_allocator_page_index = VoxelMalloc_AllocationMetadata_extract_chunk_local_allocator_page_index(allocation_metadata);
    const u32 page_bits_consumed = VoxelMalloc_AllocationMetadata_extract_page_bits_consumed(allocation_metadata);

    // The idea behind the synchronization here:
    // We fetch the page info with a single atomic operation.
    // Then we modify a local copy of the page info.
    // After we are done with our modifications we *try* to write it back.
    // To try we do an atomicCompSwap with the initialy fetched value for comparison.
    // If the values are the same this means that out changes are valid as the effect would have been as if this whole thing would have happened atomically.
    // If the values are NOT the same, another thread modified the data in a way that makes our changes INVALID.

    const u32 allocation_bits = ((1 << page_bits_consumed) - 1) << local_page_alloc_offset;

    bool deallocate_page = false;
    while (true) {
        const VoxelMalloc_PageInfo const_page_info_copy = atomicAdd(PAGE_ALLOC_INFOS(chunk_local_allocator_page_index), 0);
        const u32 page_local_consumption_bitmask = VoxelMalloc_PageInfo_extract_local_consumption_bitmask(const_page_info_copy);

        const u32 page_local_consumption_bitmask_deallocated = page_local_consumption_bitmask & ~(allocation_bits);
        deallocate_page = page_local_consumption_bitmask_deallocated == 0;

        VoxelMalloc_PageInfo our_page_info = 0;
        // For debugging
        if (!deallocate_page) {
            our_page_info = VoxelMalloc_PageInfo_pack(page_local_consumption_bitmask_deallocated, global_page_index);
        }

        const VoxelMalloc_PageInfo fetched_page_info = atomicCompSwap(PAGE_ALLOC_INFOS(chunk_local_allocator_page_index), const_page_info_copy, our_page_info);
        const bool free_successful = fetched_page_info == const_page_info_copy;
        if (free_successful) {
            break;
        }
    }

    if (deallocate_page) {
        const u32 free_stack_index = atomicAdd(deref(allocator).released_pages_stack_size, 1);
        deref(deref(allocator).released_pages_stack[free_stack_index]) = global_page_index;
    }
#endif
}
#undef PAGE_ALLOC_INFOS

daxa_RWBufferPtr(daxa_u32) voxel_malloc_address_to_base_u32_ptr(daxa_RWBufferPtr(VoxelMalloc_GlobalAllocator) allocator, VoxelMalloc_Pointer address) {
#if USE_OLD_ALLOC
    return deref(allocator).heap + address;
#else
    daxa_RWBufferPtr(VoxelMalloc_Page) page = deref(allocator).pages[VoxelMalloc_Pointer_extract_global_page_index(address)];
    daxa_RWBufferPtr(daxa_u32) result = daxa_RWBufferPtr(daxa_u32)(page);
    return result + VoxelMalloc_Pointer_extract_local_page_alloc_offset(address) * VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT;
#endif
}

daxa_RWBufferPtr(daxa_u32) voxel_malloc_address_to_u32_ptr(daxa_RWBufferPtr(VoxelMalloc_GlobalAllocator) allocator, VoxelMalloc_Pointer address) {
#if USE_OLD_ALLOC
    return deref(allocator).heap + address;
#else
    daxa_RWBufferPtr(VoxelMalloc_Page) page = deref(allocator).pages[VoxelMalloc_Pointer_extract_global_page_index(address)];
    daxa_RWBufferPtr(daxa_u32) result = daxa_RWBufferPtr(daxa_u32)(page);
    return result + (VoxelMalloc_Pointer_extract_local_page_alloc_offset(address) * VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT + 1);
#endif
}

#define INPUT deref(input_ptr)
void voxel_malloc_perframe(
    daxa_BufferPtr(GpuInput) input_ptr,
    daxa_RWBufferPtr(VoxelMalloc_GlobalAllocator) allocator) {
#if USE_OLD_ALLOC
    if (INPUT.actions[GAME_ACTION_INTERACT0] != 0) {
        deref(allocator).offset = 0;
    }
#else
    // if (INPUT.actions[GAME_ACTION_INTERACT0] != 0) {
    //     deref(allocator).page_count = 0;
    //     deref(allocator).available_pages_stack_size = 0;
    //     deref(allocator).released_pages_stack_size = 0;
    // }

    deref(allocator).available_pages_stack_size = 0;
    while (deref(allocator).released_pages_stack_size > 0) {
        --deref(allocator).released_pages_stack_size;
        deref(deref(allocator).available_pages_stack[deref(allocator).available_pages_stack_size]) = deref(deref(allocator).available_pages_stack[deref(allocator).released_pages_stack_size]);
        ++deref(allocator).available_pages_stack_size;
    }
#endif
}
#undef INPUT

// Must enter with 512 thread work group with all threads active.
void VoxelMalloc_realloc(daxa_RWBufferPtr(VoxelMalloc_GlobalAllocator) allocator, daxa_RWBufferPtr(VoxelChunk) voxel_chunk_ptr, in out VoxelMalloc_Pointer prev_address, u32 size) {
#if USE_OLD_ALLOC
#else
    u32 new_local_allocation_bit_n = (size + 1 + VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT - 1) / VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT;
    VoxelMalloc_AllocationMetadata prev_alloc_metadata = deref(voxel_malloc_address_to_base_u32_ptr(allocator, prev_address));
    u32 prev_local_allocation_bit_n = VoxelMalloc_AllocationMetadata_extract_page_bits_consumed(prev_alloc_metadata);
    if (prev_local_allocation_bit_n == new_local_allocation_bit_n) {
        return;
    }
    if (gl_LocalInvocationIndex == 0) {
        VoxelMalloc_free(allocator, voxel_chunk_ptr, prev_address);
    }
    // Is this necessary?
    barrier();
    prev_address = VoxelMalloc_malloc(allocator, voxel_chunk_ptr, size);
#endif
}

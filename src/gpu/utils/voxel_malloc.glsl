#pragma once

#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_debug_printf : require

#include <shared/shared.inl>

// See 'VoxelMalloc_PageInfo' in shared/voxel_malloc.inl
u32 VoxelMalloc_extract_page_local_consumption_bitmask(u64 page_info_bits) {
    return u32(page_info_bits >> 0) & ((1 << VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD) - 1);
}

// See 'VoxelMalloc_PageInfo' in shared/voxel_malloc.inl
u32 VoxelMalloc_extract_global_page_index(u64 page_info_bits) {
    return u32(page_info_bits >> VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD) & ((1 << 27) - 1);
}

// See 'VoxelMalloc_PageInfo' in shared/voxel_malloc.inl
VoxelMalloc_PageInfo VoxelMalloc_pack_page_info(u32 page_local_consumption_bitmask, u32 global_page_index) {
    return (u64(page_local_consumption_bitmask) << 0) | (u64(global_page_index) << VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD);
}

// See 'VoxelMalloc_Pointer' in shared/voxel_malloc.inl
//  local_page_alloc_offset must be less than
VoxelMalloc_Pointer VoxelMalloc_create_pointer(u32 global_page_index, u32 local_page_alloc_offset) {
    return (local_page_alloc_offset << 0) | (global_page_index << 5);
}

// Must enter with 512 thread work group with all threads active.
shared i32 VoxelMalloc_malloc_elected_thread;
shared i32 VoxelMalloc_malloc_elected_fallback_thread;
shared i32 VoxelMalloc_malloc_elected_fallback_page_local_consumption_bitmask;
shared bool VoxelMalloc_malloc_allocation_success;
shared u32 VoxelMalloc_malloc_global_page_index;
shared u32 VoxelMalloc_malloc_global_page_local_consumption_bitmask_first_used_bit;
shared u32 DEBUG_SHARED_VAR;
#define PAGE_ALLOC_INFOS(i) deref(voxel_chunk_ptr).sub_allocator_state.page_allocation_infos[i]
VoxelMalloc_Pointer VoxelMalloc_malloc(daxa_RWBufferPtr(VoxelMalloc_GlobalAllocator) allocator, daxa_RWBufferPtr(VoxelChunk) voxel_chunk_ptr, u32 size) {
#if USE_OLD_ALLOC
    u32 result_address = atomicAdd(deref(allocator).offset, size + 1);
    deref(deref(allocator).heap[result_address]) = size + 1;
    return result_address + 1;
#else
    u32 local_allocation_bit_n = (size + VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT - 1) / VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT;
    // u32 full_size_bytes = local_allocation_bit_n * GLOBAL_ALLOCATOR_SETTINGS.page_local_consumption_bitmask_bit_coverage;
    u32 local_allocation_bitmask_no_offset = (1 << local_allocation_bit_n) - 1;
    i32 group_local_thread_index = i32(gl_LocalInvocationIndex);
    if (group_local_thread_index == 0) {
        VoxelMalloc_malloc_elected_thread = -1;
        VoxelMalloc_malloc_elected_fallback_thread = -1;
        VoxelMalloc_malloc_allocation_success = false;
    }

    // Ask patrick if this is necessary, since above he sets these shared variables
    barrier();
    memoryBarrierShared();

    // First try to allocate into existing pages:
    bool page_unallocated = false;
    const u32 chunk_local_allocator_page_index = group_local_thread_index;
    while (!VoxelMalloc_malloc_allocation_success) {
        const VoxelMalloc_PageInfo const_page_info_copy = atomicAdd(PAGE_ALLOC_INFOS(chunk_local_allocator_page_index), 0);
        const u32 page_local_consumption_bitmask_before = VoxelMalloc_extract_page_local_consumption_bitmask(const_page_info_copy);
        const u32 global_page_index = VoxelMalloc_extract_global_page_index(const_page_info_copy);
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
                    // u32 bits_a[VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD];
                    // u32 bits_b[VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD];
                    // for (u32 i = 0; i < VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD; ++i) {
                    //     bits_a[i] = (page_local_consumption_bitmask_before >> i) & 1;
                    //     bits_b[i] = (page_local_consumption_bitmask_with_new_allocation >> i) & 1;
                    // }
                    // debugPrintfEXT("before: %x%x%x'%x%x%x%x'%x%x%x%x'%x%x%x%x'%x%x%x%x'%x%x%x%x'%x%x%x%x \nafter:  %x%x%x'%x%x%x%x'%x%x%x%x'%x%x%x%x'%x%x%x%x'%x%x%x%x'%x%x%x%x\nbit count: %i\n\n",
                    // bits_a[4 * 6 + 2], bits_a[4 * 6 + 1], bits_a[4 * 6 + 0],
                    // bits_a[4 * 5 + 3], bits_a[4 * 5 + 2], bits_a[4 * 5 + 1], bits_a[4 * 5 + 0],
                    // bits_a[4 * 4 + 3], bits_a[4 * 4 + 2], bits_a[4 * 4 + 1], bits_a[4 * 4 + 0],
                    // bits_a[4 * 3 + 3], bits_a[4 * 3 + 2], bits_a[4 * 3 + 1], bits_a[4 * 3 + 0],
                    // bits_a[4 * 2 + 3], bits_a[4 * 2 + 2], bits_a[4 * 2 + 1], bits_a[4 * 2 + 0],
                    // bits_a[4 * 1 + 3], bits_a[4 * 1 + 2], bits_a[4 * 1 + 1], bits_a[4 * 1 + 0],
                    // bits_a[4 * 0 + 3], bits_a[4 * 0 + 2], bits_a[4 * 0 + 1], bits_a[4 * 0 + 0],
                    // bits_b[4 * 6 + 2], bits_b[4 * 6 + 1], bits_b[4 * 6 + 0],
                    // bits_b[4 * 5 + 3], bits_b[4 * 5 + 2], bits_b[4 * 5 + 1], bits_b[4 * 5 + 0],
                    // bits_b[4 * 4 + 3], bits_b[4 * 4 + 2], bits_b[4 * 4 + 1], bits_b[4 * 4 + 0],
                    // bits_b[4 * 3 + 3], bits_b[4 * 3 + 2], bits_b[4 * 3 + 1], bits_b[4 * 3 + 0],
                    // bits_b[4 * 2 + 3], bits_b[4 * 2 + 2], bits_b[4 * 2 + 1], bits_b[4 * 2 + 0],
                    // bits_b[4 * 1 + 3], bits_b[4 * 1 + 2], bits_b[4 * 1 + 1], bits_b[4 * 1 + 0],
                    // bits_b[4 * 0 + 3], bits_b[4 * 0 + 2], bits_b[4 * 0 + 1], bits_b[4 * 0 + 0],
                    // local_allocation_bit_n
                    // );
                    break;
                }
            }
        }
        if (can_allocate) {
            atomicCompSwap(VoxelMalloc_malloc_elected_thread, -1, group_local_thread_index);
        }
        barrier();
        memoryBarrierShared();
        const bool no_thread_elected = VoxelMalloc_malloc_elected_thread == -1;
        // When no thread gets elected, it means that no page can be allocated into.
        // We need to break and allocate a new page in the fallback loop below.
        if (no_thread_elected) {
            // THIS IS NON DIVERGENT!
            break;
        }
        // The elected thread tries to finalize its allocation.
        else if (VoxelMalloc_malloc_elected_thread == group_local_thread_index) {
            // Try to publish the allocation into the local page.
            // Construct new page info blob.
            const u64 new_page_info = VoxelMalloc_pack_page_info(page_local_consumption_bitmask_with_new_allocation, chunk_local_allocator_page_index);
            // Try to finalize the allocation. The allocation succeses, when the fetched page info we got in the beginning did not change up until this atomic operation.
            // If the page info changed, we need to start over as we worked on outdated information for this attempt.
            const u64 fetched_page_info = atomicCompSwap(PAGE_ALLOC_INFOS(chunk_local_allocator_page_index), const_page_info_copy, new_page_info);
            VoxelMalloc_malloc_allocation_success = fetched_page_info == const_page_info_copy;
            if (VoxelMalloc_malloc_allocation_success) {
                VoxelMalloc_malloc_global_page_local_consumption_bitmask_first_used_bit = page_local_consumption_bitmask_first_used_bit;
                VoxelMalloc_malloc_global_page_index = global_page_index;
            }
        }
        barrier();
        memoryBarrierShared();
    }
    barrier();
    memoryBarrierShared();
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
            VoxelMalloc_malloc_global_page_local_consumption_bitmask_first_used_bit = 0;
            VoxelMalloc_malloc_elected_fallback_page_local_consumption_bitmask = i32(local_allocation_bitmask_no_offset);
        }
        // Then find a page meta data array element that is empty.
        // When one is found, write the metadata to it.
        barrier();
        memoryBarrierShared();
        i32 loop_iter = 0;
        while (!VoxelMalloc_malloc_allocation_success) {
            if (group_local_thread_index == 0) {
                // debugPrintfEXT("loop_iter: %i, threads online in loop: %i \n", loop_iter, DEBUG_SHARED_VAR);
                DEBUG_SHARED_VAR = 0;
            }
            barrier();
            memoryBarrierShared();

            atomicAdd(DEBUG_SHARED_VAR, page_unallocated ? 1 : 0);

            barrier();
            memoryBarrierShared();

            ++loop_iter;
            if (page_unallocated) {
                // Elect ONE of the threads that map to an empty page.
                i32 elected_thread = atomicCompSwap(VoxelMalloc_malloc_elected_thread, -1, group_local_thread_index);
                if (elected_thread == group_local_thread_index) {
                    // Pack metadata.
                    const u64 new_page_info = VoxelMalloc_pack_page_info(u32(VoxelMalloc_malloc_elected_fallback_page_local_consumption_bitmask), VoxelMalloc_malloc_global_page_index);
                    // Try to write metadata to the elected page meta info.
                    const u64 fetched_page_info = atomicCompSwap(PAGE_ALLOC_INFOS(chunk_local_allocator_page_index), u64(0), new_page_info);
                    // We succeed, when the page meta info was 0 (meaning it was empty) in the atomic comp swap.
                    // If we get back a 0, we have successfully written the page meta info.
                    VoxelMalloc_malloc_allocation_success = fetched_page_info == u64(0);
                    // debugPrintfEXT("allocation success: %i, elected_thread_index: %i, loop_iter: %i, threads online in loop: %i \n", VoxelMalloc_malloc_allocation_success, group_local_thread_index, loop_iter, DEBUG_SHARED_VAR);
                    // It can happen that another workgroup changes the metadata of our empty page, as everything is parallel here.
                    // If that happenes we need to update the boolean determining if we can update the meta data of the page the thread maps to.
                    page_unallocated = fetched_page_info == u64(0);
                    if (VoxelMalloc_malloc_allocation_success) {
                        daxa_RWBufferPtr(VoxelMalloc_Page) page = deref(allocator).pages[VoxelMalloc_malloc_global_page_index];
                        deref(page).data[VoxelMalloc_malloc_global_page_local_consumption_bitmask_first_used_bit * VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT] = (chunk_local_allocator_page_index << 0) | (local_allocation_bit_n << 9);
                    } else {
                        // Reset the thread voting value for the text allocation attempt.
                        VoxelMalloc_malloc_elected_thread = -1;
                    }
                }
            }
            barrier();
            memoryBarrierShared();
        }
    }

    return VoxelMalloc_create_pointer(VoxelMalloc_malloc_global_page_index, VoxelMalloc_malloc_global_page_local_consumption_bitmask_first_used_bit);
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
    const u32 global_page_index = address >> 5;

    // Only ONE thread should ever enter this!
    daxa_RWBufferPtr(VoxelMalloc_Page) page = deref(allocator).pages[global_page_index];
    u32 local_page_alloc_offset = address & 0x1f;

    const VoxelMalloc_AllocationMetadata allocation_metadata = deref(page).data[local_page_alloc_offset * VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT];
    const u32 chunk_local_allocator_page_index = allocation_metadata & 0x1ff;
    const u32 page_bits_consumed = allocation_metadata >> 9;

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
        const u32 page_local_consumption_bitmask = VoxelMalloc_extract_page_local_consumption_bitmask(const_page_info_copy);
        // const u32 global_page_index = VoxelMalloc_extract_global_page_index(const_page_info_copy);

        // const u32 allocation_bits = ...;
        const u32 page_local_consumption_bitmask_deallocated = page_local_consumption_bitmask & ~(allocation_bits);
        deallocate_page = page_local_consumption_bitmask_deallocated == 0;

        VoxelMalloc_PageInfo our_page_info = 0;
        // For debugging
        if (!deallocate_page) {
            our_page_info = VoxelMalloc_pack_page_info(page_local_consumption_bitmask_deallocated, global_page_index);
        }

        const VoxelMalloc_PageInfo fetched_page_info = atomicCompSwap(PAGE_ALLOC_INFOS(chunk_local_allocator_page_index), const_page_info_copy, our_page_info);
        const bool free_successfull = fetched_page_info == const_page_info_copy;
        if (free_successfull) {
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
    daxa_RWBufferPtr(VoxelMalloc_Page) page = deref(allocator).pages[address >> 5];
    daxa_RWBufferPtr(daxa_u32) result = daxa_RWBufferPtr(daxa_u32)(page);
    return result + (address & 0x1f) * VOXEL_MALLOC_MAX_ALLOCATION_SIZE_U32S;
#endif
}

daxa_RWBufferPtr(daxa_u32) voxel_malloc_address_to_u32_ptr(daxa_RWBufferPtr(VoxelMalloc_GlobalAllocator) allocator, VoxelMalloc_Pointer address) {
#if USE_OLD_ALLOC
    return deref(allocator).heap + address;
#else
    daxa_RWBufferPtr(VoxelMalloc_Page) page = deref(allocator).pages[address >> 5];
    daxa_RWBufferPtr(daxa_u32) result = daxa_RWBufferPtr(daxa_u32)(page);
    return result + (address & 0x1f) * VOXEL_MALLOC_MAX_ALLOCATION_SIZE_U32S + 1;
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
    deref(allocator).available_pages_stack_size = 0;
#endif
}
#undef INPUT

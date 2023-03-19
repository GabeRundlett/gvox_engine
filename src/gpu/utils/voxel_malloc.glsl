#pragma once

#extension GL_EXT_shader_atomic_int64 : require

#include <shared/shared.inl>

// See 'VoxelMalloc_PageInfo' in shared/voxel_malloc.inl
u32 VoxelMalloc_extract_page_local_consumption_bitmask(u64 page_info_bits) {
    return u32(page_info_bits >> 0) & ((1 << 29) - 1);
}

// See 'VoxelMalloc_PageInfo' in shared/voxel_malloc.inl
u32 VoxelMalloc_extract_global_page_index(u64 page_info_bits) {
    return u32(page_info_bits >> 29) & ((1 << 27) - 1);
}

// See 'VoxelMalloc_PageInfo' in shared/voxel_malloc.inl
VoxelMalloc_PageInfo VoxelMalloc_pack_page_info(u32 page_local_consumption_bitmask, u32 global_page_index) {
    return (u64(page_local_consumption_bitmask) << 0) | (u64(global_page_index) << 29);
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
shared i32 VoxelMalloc_malloc_elected_fallback_page_local_consumption_bitmask;
shared bool VoxelMalloc_malloc_allocation_success;
shared u32 VoxelMalloc_malloc_global_page_index;
shared u32 VoxelMalloc_malloc_global_page_local_consumption_bitmask_first_used_bit;
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
    while (true) {
        const VoxelMalloc_PageInfo const_page_info_copy = atomicAdd(PAGE_ALLOC_INFOS(chunk_local_allocator_page_index), 0);
        const u32 page_local_consumption_bitmask = VoxelMalloc_extract_page_local_consumption_bitmask(const_page_info_copy);
        const u32 global_page_index = VoxelMalloc_extract_global_page_index(const_page_info_copy);
        page_unallocated = page_local_consumption_bitmask == 0;
        bool can_allocate = false;
        u32 page_local_consumption_bitmask_first_used_bit;
        if (!page_unallocated) {
            const u32 bit_count_before = bitCount(page_local_consumption_bitmask);
            const u32 first_zero_bit_offset = findLSB(~page_local_consumption_bitmask);
            for (u32 offset = first_zero_bit_offset; offset < (29 - local_allocation_bit_n); ++offset) {
                const u32 potential_local_allocation_bitmask = page_local_consumption_bitmask | (local_allocation_bitmask_no_offset << offset);
                if (bitCount(potential_local_allocation_bitmask) == bit_count_before + local_allocation_bit_n) {
                    page_local_consumption_bitmask_first_used_bit = offset;
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
        } else if (VoxelMalloc_malloc_elected_fallback_thread == group_local_thread_index) {
            const u64 new_page_info = VoxelMalloc_pack_page_info(page_local_consumption_bitmask, chunk_local_allocator_page_index);
            const u64 fetched_page_info = atomicCompSwap(PAGE_ALLOC_INFOS(chunk_local_allocator_page_index), const_page_info_copy, new_page_info);
            VoxelMalloc_malloc_allocation_success = fetched_page_info == const_page_info_copy;
            if (VoxelMalloc_malloc_allocation_success) {
                VoxelMalloc_malloc_global_page_local_consumption_bitmask_first_used_bit = page_local_consumption_bitmask_first_used_bit;
                VoxelMalloc_malloc_global_page_index = global_page_index;
            }
        } else {
            atomicCompSwap(VoxelMalloc_malloc_elected_fallback_thread, -1, group_local_thread_index);
        }
        if (VoxelMalloc_malloc_allocation_success) {
            break;
        }
    }
    barrier();
    memoryBarrierShared();
    // If allocating into existing pages fails because all current pages have too little space,
    // allocate a new page and set one of the free page metadatas to contain it.
    if (!VoxelMalloc_malloc_allocation_success) {
        // First allocate the new page and calculate the metadata for it.
        if (group_local_thread_index == 0) {
            // TODO: Fix this. What if the stack size is zero?
            i32 global_page_stack_index = atomicAdd(deref(allocator).available_pages_stack_size, -1);
            VoxelMalloc_malloc_global_page_local_consumption_bitmask_first_used_bit = 0;
            VoxelMalloc_malloc_elected_fallback_page_local_consumption_bitmask = i32(local_allocation_bitmask_no_offset);
            if (global_page_stack_index < 0) {
                atomicAdd(deref(allocator).available_pages_stack_size, 1);
                atomicAdd(deref(allocator).page_count, 1);
            } else {
                VoxelMalloc_malloc_elected_fallback_page_index = i32(deref(deref(allocator).available_pages_stack[global_page_stack_index]));
            }
        }
        // Then find a page meta data array element that is empty.
        // When one is found, write the metadata to it.
        barrier();
        memoryBarrierShared();
        while (true) {
            if (page_unallocated) {
                // Elect ONE of the threads that map to an empty page.
                i32 elected_thread = atomicCompSwap(VoxelMalloc_malloc_elected_thread, -1, group_local_thread_index);
                if (elected_thread == group_local_thread_index) {
                    // Pack metadata.
                    const u64 new_page_info = VoxelMalloc_pack_page_info(u32(VoxelMalloc_malloc_elected_fallback_page_local_consumption_bitmask), VoxelMalloc_malloc_elected_fallback_page_index);
                    // Try to write metadata to the elected page meta info.
                    const u64 fetched_page_info = atomicCompSwap(PAGE_ALLOC_INFOS(chunk_local_allocator_page_index), u64(0), new_page_info);
                    // We succeed, when the page meta info was 0 (meaning it was empty) in the atomic comp swap.
                    // If we get back a 0, we have successfully written the page meta info.
                    VoxelMalloc_malloc_allocation_success = fetched_page_info == u64(0);
                    // It can happen that another workgroup changes the metadata of our empty page, as everything is parallel here.
                    // If that happenes we need to update the boolean determining if we can update the meta data of the page the thread maps to.
                    page_unallocated = fetched_page_info == u64(0);
                    if (VoxelMalloc_malloc_allocation_success) {
                        daxa_RWBufferPtr(VoxelMalloc_Page) page = deref(allocator).pages[VoxelMalloc_malloc_elected_fallback_page_index];
                        deref(page).data[VoxelMalloc_malloc_global_page_local_consumption_bitmask_first_used_bit * VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT] = (chunk_local_allocator_page_index << 0) | (local_allocation_bit_n << 9);
                    }
                }
            }
            memoryBarrierShared();
            if (VoxelMalloc_malloc_allocation_success) {
                break;
            }
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

daxa_RWBufferPtr(daxa_u32) voxel_malloc_address_to_u32_ptr(daxa_RWBufferPtr(VoxelMalloc_GlobalAllocator) allocator, VoxelMalloc_Pointer address) {
#if USE_OLD_ALLOC
    return deref(allocator).heap + address;
#else
    daxa_RWBufferPtr(VoxelMalloc_Page) page = deref(allocator).pages[address >> 5];
    daxa_RWBufferPtr(daxa_u32) result = daxa_RWBufferPtr(daxa_u32)(page);
    return result + (address & 0x1f) * VOXEL_MALLOC_MAX_ALLOCATION_SIZE_U32S;
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
#endif
}
#undef INPUT

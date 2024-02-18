#pragma once

#extension GL_EXT_shader_atomic_int64 : require

#define UserAllocatorType VoxelMallocPageAllocator
#define UserIndexType VoxelMalloc_PageIndex
#include <utilities/allocator.glsl>

// See 'VoxelMalloc_PageInfo' in shared/voxel_malloc.inl
uint VoxelMalloc_PageInfo_extract_local_consumption_bitmask(VoxelMalloc_PageInfo page_info_bits) {
#if VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD == 32
    return uint(page_info_bits);
#else
    return uint(page_info_bits >> 0) & ((1 << VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD) - 1);
#endif
}
uint VoxelMalloc_PageInfo_extract_global_page_index(VoxelMalloc_PageInfo page_info_bits) {
    return uint(page_info_bits >> VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD) & ((1 << VOXEL_MALLOC_LOG2_MAX_GLOBAL_PAGE_COUNT) - 1);
}
VoxelMalloc_PageInfo VoxelMalloc_PageInfo_pack(uint page_local_consumption_bitmask, uint global_page_index) {
    return (daxa_u64(page_local_consumption_bitmask) << 0) | (daxa_u64(global_page_index) << VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD);
}

// See 'VoxelMalloc_AllocationMetadata' in shared/voxel_malloc.inl
uint VoxelMalloc_AllocationMetadata_extract_chunk_local_allocator_page_index(VoxelMalloc_AllocationMetadata metadata) {
    return (metadata >> 0) & 0x1ff;
}
uint VoxelMalloc_AllocationMetadata_extract_page_bits_consumed(VoxelMalloc_AllocationMetadata metadata) {
    return (metadata >> 9);
}
VoxelMalloc_AllocationMetadata VoxelMalloc_AllocationMetadata_pack(uint chunk_local_allocator_page_index, uint page_bits_consumed) {
    return (chunk_local_allocator_page_index << 0) | (page_bits_consumed << 9);
}

// See 'VoxelMalloc_Pointer' in shared/voxel_malloc.inl
uint VoxelMalloc_Pointer_extract_local_page_alloc_offset(VoxelMalloc_Pointer ptr) {
    return (ptr >> 0) & 0x1f;
}
uint VoxelMalloc_Pointer_extract_global_page_index(VoxelMalloc_Pointer ptr) {
    return (ptr >> 5);
}
VoxelMalloc_Pointer VoxelMalloc_Pointer_pack(uint global_page_index, uint local_page_alloc_offset) {
    return (local_page_alloc_offset << 0) | (global_page_index << 5);
}

void voxel_malloc_address_to_base_u32_ptr(daxa_BufferPtr(VoxelMallocPageAllocator) allocator, VoxelMalloc_Pointer address, out daxa_RWBufferPtr(uint) result) {
    daxa_RWBufferPtr(uint) page = advance(deref(allocator).heap, VoxelMalloc_Pointer_extract_global_page_index(address) * VOXEL_MALLOC_PAGE_SIZE_U32S);
    result = advance(page, VoxelMalloc_Pointer_extract_local_page_alloc_offset(address) * VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT);
}

void voxel_malloc_address_to_u32_ptr(daxa_BufferPtr(VoxelMallocPageAllocator) allocator, VoxelMalloc_Pointer address, out daxa_RWBufferPtr(uint) result) {
    daxa_RWBufferPtr(uint) page = advance(deref(allocator).heap, VoxelMalloc_Pointer_extract_global_page_index(address) * VOXEL_MALLOC_PAGE_SIZE_U32S);
    result = advance(page, (VoxelMalloc_Pointer_extract_local_page_alloc_offset(address) * VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT + 1));
}

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_COMPUTE
// Must enter with 512 thread work group with all threads active.
shared int VoxelMalloc_malloc_elected_fallback_thread;
shared int VoxelMalloc_malloc_elected_fallback_page_local_consumption_bitmask;
shared bool VoxelMalloc_malloc_allocation_success;
shared uint VoxelMalloc_malloc_global_page_index;
shared uint VoxelMalloc_malloc_page_local_consumption_bitmask_first_used_bit;
// We need three of these variables.
// We read from the last iteration's variable, we write to the current iteration's variable and we reset the next iteration's variable.
// To perform all these actions with just one barrier, we must use three distinct variables for each action to avoid race conditions.
// As all threads run asynchronously between barriers, we can only ever do one of these things
// at a time as this would cause race conditions.
shared int VoxelMalloc_malloc_elected_thread[3];
#define PAGE_ALLOC_INFOS(i) deref(voxel_chunk_ptr).sub_allocator_state.page_allocation_infos[i]
VoxelMalloc_Pointer VoxelMalloc_malloc(daxa_RWBufferPtr(VoxelMallocPageAllocator) allocator, daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunk_ptr, uint size) {
    uint local_allocation_bit_n = (size + 1 + VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT - 1) / VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT;
    uint local_allocation_bitmask_no_offset = (1 << local_allocation_bit_n) - 1;
    int group_local_thread_index = int(gl_LocalInvocationIndex);
    if (group_local_thread_index == 0) {
        VoxelMalloc_malloc_elected_thread[0] = -1;
        VoxelMalloc_malloc_elected_thread[1] = -1;
        VoxelMalloc_malloc_elected_thread[2] = -1;
        VoxelMalloc_malloc_elected_fallback_thread = -1;
        VoxelMalloc_malloc_allocation_success = false;
    }

    // First try to allocate into existing pages:
    bool page_unallocated = false;
    bool did_it_already = false;
    bool did_it_already1 = false;
    barrier();
    memoryBarrierShared();
    const uint chunk_local_allocator_page_index = group_local_thread_index;
    for (uint iteration = 0; true; ++iteration) {
        const uint current_election_variable = iteration % 3;
        const uint prev_iteration_election_variable = (iteration + 2) % 3;
        const VoxelMalloc_PageInfo const_page_info_copy = atomicAdd(PAGE_ALLOC_INFOS(chunk_local_allocator_page_index), 0);
        const uint page_local_consumption_bitmask_before = VoxelMalloc_PageInfo_extract_local_consumption_bitmask(const_page_info_copy);
        const uint global_page_index = VoxelMalloc_PageInfo_extract_global_page_index(const_page_info_copy);
        page_unallocated = page_local_consumption_bitmask_before == 0;
        bool can_allocate = false;
        uint page_local_consumption_bitmask_first_used_bit;
        uint page_local_consumption_bitmask_with_new_allocation = 0;
        if (!page_unallocated) {
            const uint bit_count_before = bitCount(page_local_consumption_bitmask_before);
            const uint first_zero_bit_offset = findLSB(~page_local_consumption_bitmask_before);
            for (uint offset = first_zero_bit_offset; offset < (VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD - local_allocation_bit_n) + 1; ++offset) {
                const uint potential_local_allocation_bitmask = page_local_consumption_bitmask_before | (local_allocation_bitmask_no_offset << offset);
                if (bitCount(potential_local_allocation_bitmask) == bit_count_before + local_allocation_bit_n) {
                    page_local_consumption_bitmask_first_used_bit = offset;
                    page_local_consumption_bitmask_with_new_allocation = potential_local_allocation_bitmask;
                    can_allocate = true;
                    break;
                }
            }
        }
        if (can_allocate) {
            int election_fetch = atomicCompSwap(VoxelMalloc_malloc_elected_thread[current_election_variable], -1, group_local_thread_index);
            if (election_fetch == -1) {
                // Thread won election.
                // Try to publish the allocation into the local page.
                // Construct new page info blob.
                const daxa_u64 new_page_info = VoxelMalloc_PageInfo_pack(page_local_consumption_bitmask_with_new_allocation, global_page_index);
                // Try to finalize the allocation. The allocation succeses, when the fetched page
                // info we got in the beginning did not change up until this atomic operation.
                // If the page info changed, we need to start over as we worked on outdated information for this attempt.
                const daxa_u64 fetched_page_info = atomicCompSwap(PAGE_ALLOC_INFOS(chunk_local_allocator_page_index), const_page_info_copy, new_page_info);
                VoxelMalloc_malloc_allocation_success = (fetched_page_info == const_page_info_copy);
                if (VoxelMalloc_malloc_allocation_success) {
                    VoxelMalloc_malloc_page_local_consumption_bitmask_first_used_bit = page_local_consumption_bitmask_first_used_bit;
                    VoxelMalloc_malloc_global_page_index = global_page_index;
                    daxa_RWBufferPtr(uint) page = advance(deref(allocator).heap, VoxelMalloc_malloc_global_page_index * VOXEL_MALLOC_PAGE_SIZE_U32S);
                    deref(advance(page, page_local_consumption_bitmask_first_used_bit * VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT)) = VoxelMalloc_AllocationMetadata_pack(chunk_local_allocator_page_index, local_allocation_bit_n);
                }
            }
        }
        barrier();
        memoryBarrierShared();
        // Allocation is impossible when all threads were unable to be elected for the allocation.
        // In this case we break and use the fallback loop below to allocate a completely new page, we write out result into.`
        const bool allocation_impossible = (VoxelMalloc_malloc_elected_thread[current_election_variable] == -1) && (iteration != 0);
        const bool uniform_breaking_condition = allocation_impossible || VoxelMalloc_malloc_allocation_success;
        if (uniform_breaking_condition) {
            // THIS IS NON DIVERGENT!
            break;
        }
        if (group_local_thread_index == 0) {
            // Reset the election variable of the previous iteration.
            // We can do this safely in here, as the current and the next iterations election variables are the only ones that are used right now.
            VoxelMalloc_malloc_elected_thread[prev_iteration_election_variable] = -1;
        }
    }
    // If allocating into existing pages fails because all current pages have too little space,
    // allocate a new page and set one of the free page metadatas to contain it.
    if (!VoxelMalloc_malloc_allocation_success) {
        // First allocate the new page and calculate the metadata for it.
        if (group_local_thread_index == 0) {
            VoxelMalloc_malloc_global_page_index = VoxelMallocPageAllocator_malloc(allocator);
            VoxelMalloc_malloc_page_local_consumption_bitmask_first_used_bit = 0;
        }
        // Then find a page meta data array element that is empty.
        // When one is found, write the metadata to it.
        barrier();
        memoryBarrierShared();
        for (uint iteration = 0; !VoxelMalloc_malloc_allocation_success; ++iteration) {
            const uint current_election_variable = iteration % 2;
            const uint next_iteration_election_variable = (iteration + 1) % 2;
            if (group_local_thread_index == 0) {
                VoxelMalloc_malloc_elected_thread[next_iteration_election_variable] = -1;
            }
            if (page_unallocated) {
                // Elect ONE of the threads that map to an empty page.
                int election_fetch = atomicCompSwap(VoxelMalloc_malloc_elected_thread[current_election_variable], -1, group_local_thread_index);
                if (election_fetch == -1) {
                    // Pack metadata.
                    const daxa_u64 new_page_info = VoxelMalloc_PageInfo_pack(local_allocation_bitmask_no_offset, VoxelMalloc_malloc_global_page_index);
                    // Try to write metadata to the elected page meta info.
                    const daxa_u64 fetched_page_info = atomicCompSwap(PAGE_ALLOC_INFOS(chunk_local_allocator_page_index), daxa_u64(0), new_page_info);
                    // We succeed, when the page meta info was 0 (meaning it was empty) in the atomic comp swap.
                    // If we get back a 0, we have successfully written the page meta info.
                    VoxelMalloc_malloc_allocation_success = fetched_page_info == daxa_u64(0);
                    // It can happen that another workgroup changes the metadata of our empty page, as everything is parallel here.
                    // If that happenes we need to update the boolean determining if we can update the meta data of the page the thread maps to.
                    page_unallocated = fetched_page_info == daxa_u64(0);
                    if (VoxelMalloc_malloc_allocation_success) {
                        daxa_RWBufferPtr(uint) page = advance(deref(allocator).heap, VoxelMalloc_malloc_global_page_index * VOXEL_MALLOC_PAGE_SIZE_U32S);
                        deref(advance(page, 0)) = VoxelMalloc_AllocationMetadata_pack(chunk_local_allocator_page_index, local_allocation_bit_n);
                    }
                }
            }
            barrier();
            memoryBarrierShared();
        }
    }

    barrier();
    memoryBarrierShared();

    VoxelMalloc_Pointer blob_ptr = VoxelMalloc_Pointer_pack(VoxelMalloc_malloc_global_page_index, VoxelMalloc_malloc_page_local_consumption_bitmask_first_used_bit);
    return blob_ptr;
}
#undef PAGE_ALLOC_INFOS

// Can enter with any workgroup and warp configuration.
#define PAGE_ALLOC_INFOS(i) deref(voxel_chunk_ptr).sub_allocator_state.page_allocation_infos[i]
void VoxelMalloc_free(daxa_RWBufferPtr(VoxelMallocPageAllocator) allocator, daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunk_ptr, VoxelMalloc_Pointer address) {
    const uint global_page_index = VoxelMalloc_Pointer_extract_global_page_index(address);
    uint local_page_alloc_offset = VoxelMalloc_Pointer_extract_local_page_alloc_offset(address);

    // Only ONE thread should ever enter this!
    daxa_RWBufferPtr(uint) page = advance(deref(allocator).heap, global_page_index * VOXEL_MALLOC_PAGE_SIZE_U32S);

    const VoxelMalloc_AllocationMetadata allocation_metadata = deref(advance(page, local_page_alloc_offset * VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT));
    const uint chunk_local_allocator_page_index = VoxelMalloc_AllocationMetadata_extract_chunk_local_allocator_page_index(allocation_metadata);
    const uint page_bits_consumed = VoxelMalloc_AllocationMetadata_extract_page_bits_consumed(allocation_metadata);

    // The idea behind the synchronization here:
    // We fetch the page info with a single atomic operation.
    // Then we modify a local copy of the page info.
    // After we are done with our modifications we *try* to write it back.
    // To try we do an atomicCompSwap with the initialy fetched value for comparison.
    // If the values are the same this means that out changes are valid as the effect would have been as if this whole thing would have happened atomically.
    // If the values are NOT the same, another thread modified the data in a way that makes our changes INVALID.

    const uint allocation_bits = ((1 << page_bits_consumed) - 1) << local_page_alloc_offset;

    bool deallocate_page = false;
    while (true) {
        const VoxelMalloc_PageInfo const_page_info_copy = atomicAdd(PAGE_ALLOC_INFOS(chunk_local_allocator_page_index), 0);
        const uint page_local_consumption_bitmask = VoxelMalloc_PageInfo_extract_local_consumption_bitmask(const_page_info_copy);

        const uint page_local_consumption_bitmask_deallocated = page_local_consumption_bitmask & ~(allocation_bits);
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
        VoxelMallocPageAllocator_free(allocator, global_page_index);
    }
}
#undef PAGE_ALLOC_INFOS

// Must enter with 512 thread work group with all threads active.
void VoxelMalloc_realloc(daxa_RWBufferPtr(VoxelMallocPageAllocator) allocator, daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunk_ptr, in out VoxelMalloc_Pointer prev_address, uint size) {
    uint new_local_allocation_bit_n = (size + 1 + VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT - 1) / VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT;
    daxa_RWBufferPtr(uint) temp_ptr;
    voxel_malloc_address_to_base_u32_ptr(daxa_BufferPtr(VoxelMallocPageAllocator)(as_address(allocator)), prev_address, temp_ptr);
    VoxelMalloc_AllocationMetadata prev_alloc_metadata = deref(temp_ptr);
    uint prev_local_allocation_bit_n = VoxelMalloc_AllocationMetadata_extract_page_bits_consumed(prev_alloc_metadata);
    if (prev_local_allocation_bit_n == new_local_allocation_bit_n) {
        return;
    }
    if (gl_LocalInvocationIndex == 0) {
        VoxelMalloc_free(allocator, voxel_chunk_ptr, prev_address);
    }
    // Is this necessary?
    barrier();
    prev_address = VoxelMalloc_malloc(allocator, voxel_chunk_ptr, size);
}
#endif

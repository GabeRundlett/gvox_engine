#pragma once

#include <shared/app.inl>

#include <utils/math.glsl>
#include <voxels/impl/voxels.glsl>

void voxel_world_startup(
    daxa_RWBufferPtr(GpuGlobals) globals_ptr) {
    deref(globals_ptr).voxel_world.chunk_update_n = 0;
}

#define VOXEL_PERFRAME_INFO_PTRS voxel_malloc_page_allocator

void voxel_world_perframe(
    daxa_BufferPtr(GpuInput) gpu_input,
    daxa_BufferPtr(GpuOutput) gpu_output,
    daxa_RWBufferPtr(GpuGlobals) globals_ptr,
    daxa_RWBufferPtr(VoxelMallocPageAllocator) voxel_malloc_page_allocator) {

    for (u32 i = 0; i < MAX_CHUNK_UPDATES_PER_FRAME; ++i) {
        deref(globals_ptr).voxel_world.chunk_update_infos[i].brush_flags = 0;
        deref(globals_ptr).voxel_world.chunk_update_infos[i].i = INVALID_CHUNK_I;
    }

    deref(globals_ptr).voxel_world.chunk_update_n = 0;

    deref(globals_ptr).indirect_dispatch.chunk_edit_dispatch = u32vec3(CHUNK_SIZE / 8, CHUNK_SIZE / 8, 0);
    deref(globals_ptr).indirect_dispatch.subchunk_x2x4_dispatch = u32vec3(1, 64, 0);
    deref(globals_ptr).indirect_dispatch.subchunk_x8up_dispatch = u32vec3(1, 1, 0);

    VoxelMallocPageAllocator_perframe(voxel_malloc_page_allocator);
    // VoxelLeafChunkAllocator_perframe(voxel_leaf_chunk_allocator);
    // VoxelParentChunkAllocator_perframe(voxel_parent_chunk_allocator);

    deref(gpu_output[deref(gpu_input).fif_index]).voxel_malloc_output.current_element_count =
        VoxelMallocPageAllocator_get_consumed_element_count(daxa_BufferPtr(VoxelMallocPageAllocator)(voxel_malloc_page_allocator));
    // deref(gpu_output[deref(gpu_input).fif_index]).voxel_leaf_chunk_output.current_element_count =
    //     VoxelLeafChunkAllocator_get_consumed_element_count(voxel_leaf_chunk_allocator);
    // deref(gpu_output[deref(gpu_input).fif_index]).voxel_parent_chunk_output.current_element_count =
    //     VoxelParentChunkAllocator_get_consumed_element_count(voxel_parent_chunk_allocator);
}

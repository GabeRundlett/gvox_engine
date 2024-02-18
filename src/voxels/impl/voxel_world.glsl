#pragma once

#include <utilities/gpu/math.glsl>
#include <voxels/impl/voxels.glsl>

void voxel_world_startup(daxa_RWBufferPtr(GpuGlobals) globals_ptr, VoxelRWBufferPtrs ptrs) {
    deref(ptrs.globals).chunk_update_n = 0;
}

// #define UserAllocatorType VoxelLeafChunkAllocator
// #define UserIndexType uint
// #include <utilities/allocator.glsl>

// #define UserAllocatorType VoxelParentChunkAllocator
// #define UserIndexType uint
// #include <utilities/allocator.glsl>

// Queue a L0 terrain generation item
// void queue_terrain_generation_work_item(ivec3 chunk_offset) {
//     ChunkWorkItem terrain_work_item;
//     terrain_work_item.i = ivec3(0);
//     terrain_work_item.chunk_offset = chunk_offset;
//     terrain_work_item.brush_id = BRUSH_FLAGS_WORLD_BRUSH;
//     zero_work_item_children(terrain_work_item);
//     queue_root_work_item(globals, terrain_work_item);
// }

void voxel_world_perframe(
    daxa_BufferPtr(GpuInput) gpu_input,
    daxa_RWBufferPtr(GpuOutput) gpu_output,
    daxa_RWBufferPtr(GpuGlobals) globals_ptr,
    VoxelRWBufferPtrs ptrs) {

    for (uint i = 0; i < MAX_CHUNK_UPDATES_PER_FRAME; ++i) {
        deref(ptrs.globals).chunk_update_infos[i].brush_flags = 0;
        deref(ptrs.globals).chunk_update_infos[i].i = INVALID_CHUNK_I;
    }

    deref(ptrs.globals).chunk_update_n = 0;

    deref(ptrs.globals).prev_offset = deref(ptrs.globals).offset;
    deref(ptrs.globals).offset = deref(gpu_input).player.player_unit_offset;

    deref(globals_ptr).indirect_dispatch.chunk_edit_dispatch = uvec3(CHUNK_SIZE / 8, CHUNK_SIZE / 8, 0);
    deref(globals_ptr).indirect_dispatch.subchunk_x2x4_dispatch = uvec3(1, 64, 0);
    deref(globals_ptr).indirect_dispatch.subchunk_x8up_dispatch = uvec3(1, 1, 0);

    VoxelMallocPageAllocator_perframe(ptrs.allocator);
    // VoxelLeafChunkAllocator_perframe(ptrs.voxel_leaf_chunk_allocator);
    // VoxelParentChunkAllocator_perframe(ptrs.voxel_parent_chunk_allocator);

    deref(advance(gpu_output, deref(gpu_input).fif_index)).voxel_world.voxel_malloc_output.current_element_count =
        VoxelMallocPageAllocator_get_consumed_element_count(daxa_BufferPtr(VoxelMallocPageAllocator)(as_address(ptrs.allocator)));
    // deref(advance(gpu_output, deref(gpu_input).fif_index)).voxel_world.voxel_leaf_chunk_output.current_element_count =
    //     VoxelLeafChunkAllocator_get_consumed_element_count(voxel_leaf_chunk_allocator);
    // deref(advance(gpu_output, deref(gpu_input).fif_index)).voxel_world.voxel_parent_chunk_output.current_element_count =
    //     VoxelParentChunkAllocator_get_consumed_element_count(voxel_parent_chunk_allocator);
}

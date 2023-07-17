#pragma once

#include <shared/shared.inl>

#include <utils/math.glsl>
#include <utils/voxels.glsl>

#define PLAYER deref(globals_ptr).player
#define VOXEL_WORLD deref(globals_ptr).voxel_world
#define CHUNKS(i) deref(voxel_chunks_ptr + i)
void voxel_world_startup(
    daxa_RWBufferPtr(GpuGlobals) globals_ptr,
    daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunks_ptr) {

    VOXEL_WORLD.chunk_update_n = 0;
}
#undef CHUNKS
#undef VOXEL_WORLD
#undef PLAYER

#define SETTINGS deref(settings_ptr)
#define INPUT deref(input_ptr)
#define PLAYER deref(globals_ptr).player
#define VOXEL_WORLD deref(globals_ptr).voxel_world
#define INDIRECT deref(globals_ptr).indirect_dispatch
void voxel_world_perframe(
    daxa_BufferPtr(GpuSettings) settings_ptr,
    daxa_BufferPtr(GpuInput) input_ptr,
    daxa_RWBufferPtr(GpuGlobals) globals_ptr) {

    for (u32 i = 0; i < MAX_CHUNK_UPDATES_PER_FRAME; ++i) {
        VOXEL_WORLD.chunk_update_infos[i].flags = 0;
    }

    VOXEL_WORLD.chunk_update_n = 0;

    INDIRECT.chunk_edit_dispatch = u32vec3(CHUNK_SIZE / 8, CHUNK_SIZE / 8, 0);
    INDIRECT.subchunk_x2x4_dispatch = u32vec3(1, 64, 0);
    INDIRECT.subchunk_x8up_dispatch = u32vec3(1, 1, 0);
}
#undef INDIRECT
#undef VOXEL_WORLD
#undef PLAYER
#undef INPUT
#undef SETTINGS

// Add a root work item to the L0 ChunkWorkItems queue of the ChunkThreadPoolState
#define A_THREAD_POOL deref(globals).chunk_thread_pool_state
bool queue_root_work_item(daxa_RWBufferPtr(GpuGlobals) globals_ptr, in ChunkWorkItem new_work_item) {
    // Get the insertion index ( = number of uncompleted work items left)
    // Also increment the number of uncompleted work items
    u32 queue_offset = atomicAdd(A_THREAD_POOL.work_items_l0_uncompleted, 1);
    // Clamp the uncompleted item count to the maximum L0 work items authorized
    atomicMin(A_THREAD_POOL.work_items_l0_uncompleted, MAX_CHUNK_WORK_ITEMS_L0);
    // Check if the insertion index is valid
    if (queue_offset < MAX_CHUNK_WORK_ITEMS_L0) {
        // Insert the work item in the correct queue at the correct offset
        A_THREAD_POOL.chunk_work_items_l0[1 - A_THREAD_POOL.queue_index][queue_offset] = new_work_item;
        return true;
    } else {
        return false;
    }
}
#undef A_THREAD_POOL

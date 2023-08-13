#pragma once

#include <shared/app.inl>

#include <utils/math.glsl>
#include <voxels/voxels.glsl>

void voxel_world_startup(
    daxa_RWBufferPtr(GpuGlobals) globals_ptr) {
    deref(globals_ptr).voxel_world.chunk_update_n = 0;
}

void voxel_world_perframe(
    daxa_BufferPtr(GpuInput) input_ptr,
    daxa_RWBufferPtr(GpuGlobals) globals_ptr) {

    for (u32 i = 0; i < MAX_CHUNK_UPDATES_PER_FRAME; ++i) {
        deref(globals_ptr).voxel_world.chunk_update_infos[i].flags = 0;
    }

    deref(globals_ptr).voxel_world.chunk_update_n = 0;

    deref(globals_ptr).indirect_dispatch.chunk_edit_dispatch = u32vec3(CHUNK_SIZE / 8, CHUNK_SIZE / 8, 0);
    deref(globals_ptr).indirect_dispatch.subchunk_x2x4_dispatch = u32vec3(1, 64, 0);
    deref(globals_ptr).indirect_dispatch.subchunk_x8up_dispatch = u32vec3(1, 1, 0);
}

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

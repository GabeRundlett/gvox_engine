#include <utils/player.glsl>
#include <utils/voxel_world.glsl>

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    player_startup(globals);
    voxel_world_startup(globals, voxel_chunks);

    // TEMP: init thread pool
#define THREAD_POOL deref(globals).chunk_thread_pool_state
    for (u32 i = 0; i < MAX_NODE_THREADS; ++i) {
        THREAD_POOL.available_threads_queue[i] = i;
    }
    THREAD_POOL.job_counters_packed = u64(MAX_NODE_THREADS) << 0x00;

    ChunkNodeWorkItem new_work_item;
    new_work_item.i = 0;
    new_work_item.flags = CHUNK_WORK_FLAG_IS_READY_BIT;

    u32 work_index = u32(atomicAdd(THREAD_POOL.job_counters_packed, u64(1) << 0x20)) & (MAX_NODE_THREADS - 1);
    THREAD_POOL.chunk_node_work_items[work_index] = new_work_item;
}

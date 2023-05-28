#include <utils/player.glsl>
#include <utils/voxel_world.glsl>

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    player_startup(globals);
    voxel_world_startup(globals, voxel_chunks);

    // start world-gen by issuing a chunk update!
#define THREAD_POOL deref(globals).chunk_thread_pool_state
    THREAD_POOL.chunk_work_items_l0[0].packed_coordinate = 0;
    THREAD_POOL.chunk_work_items_l0[0].brush_id = 0;
    zero_work_item_children(THREAD_POOL.chunk_work_items_l0[0]);
    THREAD_POOL.work_items_l0_begin = 0;
    THREAD_POOL.work_items_l0_queued = 1;

    THREAD_POOL.work_items_l0_dispatch_y = 1;
    THREAD_POOL.work_items_l0_dispatch_z = 1;

    THREAD_POOL.work_items_l1_dispatch_y = 1;
    THREAD_POOL.work_items_l1_dispatch_z = 1;
}

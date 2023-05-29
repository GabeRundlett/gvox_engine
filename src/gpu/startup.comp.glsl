#include <utils/player.glsl>
#include <utils/voxel_world.glsl>

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    player_startup(globals);
    voxel_world_startup(globals, voxel_chunks);

    // start world-gen by issuing a chunk update!
    ChunkWorkItem terrain_work_item;
    terrain_work_item.i = u32vec3(0);
    terrain_work_item.brush_id = CHUNK_STAGE_WORLD_BRUSH;
    zero_work_item_children(terrain_work_item);
    queue_root_work_item(globals, terrain_work_item);

    deref(globals).chunk_thread_pool_state.work_items_l0_dispatch_y = 1;
    deref(globals).chunk_thread_pool_state.work_items_l0_dispatch_z = 1;

    deref(globals).chunk_thread_pool_state.work_items_l1_dispatch_y = 1;
    deref(globals).chunk_thread_pool_state.work_items_l1_dispatch_z = 1;
}

#include <utils/player.glsl>
#include <utils/voxel_world.glsl>

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    player_startup(gpu_input, globals);
    voxel_world_startup(globals, voxel_chunks);

    deref(globals).chunk_thread_pool_state.work_items_l0_dispatch_y = 1;
    deref(globals).chunk_thread_pool_state.work_items_l0_dispatch_z = 1;

    deref(globals).chunk_thread_pool_state.work_items_l1_dispatch_y = 1;
    deref(globals).chunk_thread_pool_state.work_items_l1_dispatch_z = 1;
}

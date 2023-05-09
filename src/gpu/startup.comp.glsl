#include <utils/player.glsl>
#include <utils/voxel_world.glsl>

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    player_startup(globals);
    voxel_world_startup(globals, voxel_chunks);
}

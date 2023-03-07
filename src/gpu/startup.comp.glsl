#include <utils/player.glsl>
#include <utils/voxel_world.glsl>

DAXA_USE_PUSH_CONSTANT(StartupComputePush)

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    player_startup(
        daxa_push_constant.gpu_settings,
        daxa_push_constant.gpu_globals);
    voxel_world_startup(
        daxa_push_constant.gpu_settings,
        daxa_push_constant.gpu_globals,
        daxa_push_constant.voxel_chunks);
}

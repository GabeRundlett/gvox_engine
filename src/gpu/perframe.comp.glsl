#include <utils/player.glsl>
#include <utils/voxel_world.glsl>

DAXA_USE_PUSH_CONSTANT(PerframeComputePush)

#define INPUT deref(daxa_push_constant.gpu_input)
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    player_perframe(
        daxa_push_constant.gpu_settings,
        daxa_push_constant.gpu_input,
        daxa_push_constant.gpu_globals);
    voxel_world_perframe(
        daxa_push_constant.gpu_settings,
        daxa_push_constant.gpu_input,
        daxa_push_constant.gpu_globals,
        daxa_push_constant.voxel_chunks);

    if (INPUT.actions[GAME_ACTION_INTERACT0] != 0) {
        deref(daxa_push_constant.gpu_allocator.state).offset = 0;
    }
}
#undef INPUT

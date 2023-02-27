#include <utils/player.glsl>

DAXA_USE_PUSH_CONSTANT(PerframeComputePush)

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    player_perframe(daxa_push_constant.gpu_settings, daxa_push_constant.gpu_input, daxa_push_constant.gpu_globals);
}

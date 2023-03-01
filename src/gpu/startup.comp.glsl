#include <shared/shared.inl>

#include <utils/math.glsl>
#include <utils/player.glsl>

DAXA_USE_PUSH_CONSTANT(StartupComputePush)

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    player_startup(daxa_push_constant.gpu_globals);
}

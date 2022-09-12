#version 450

#include <shared.inl>

DAXA_USE_PUSH_CONSTANT(StartupCompPush)

void startup_player(inout Player player) {
    player.pos = f32vec3(0, 0, 0);
    player.vel = f32vec3(0, 0, 0);
    player.rot = f32vec3(0, 0, 0);
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    startup_player(push_constant.gpu_globals.player);
}

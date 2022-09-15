#version 450

#include <shared.inl>
#include <utils/voxel.glsl>

DAXA_USE_PUSH_CONSTANT(StartupCompPush)

void startup(inout Player player) {
    player.pos = f32vec3(0, -5, 0);
    player.vel = f32vec3(0, 0, 0);
    player.rot = f32vec3(0, 0, 0);
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    startup(push_constant.gpu_globals.player);

    CHUNK.box.bound_min = f32vec3(-2, -2, -2);
    CHUNK.box.bound_max = f32vec3(+2, +2, +2);

    SCENE.sphere_n = 1;
    SCENE.spheres[0].o = f32vec3(-1, -1, -1);
    SCENE.spheres[0].r = 2;

    SCENE.box_n = 0;
}

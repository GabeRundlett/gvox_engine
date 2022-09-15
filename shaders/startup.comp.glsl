#version 450

#include <shared/shared.inl>
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

    for (i32 zi = 0; zi < CHUNK_NZ; ++zi) {
        for (i32 yi = 0; yi < CHUNK_NY; ++yi) {
            for (i32 xi = 0; xi < CHUNK_NX; ++xi) {
                u32 index = get_chunk_index(i32vec3(xi, yi, zi));
                VOXEL_CHUNKS[index].box.bound_min = f32vec3(xi, yi, zi) * (CHUNK_SIZE / VOXEL_SCL);
                VOXEL_CHUNKS[index].box.bound_max = VOXEL_CHUNKS[index].box.bound_min + (CHUNK_SIZE / VOXEL_SCL);
            }
        }
    }

    VOXEL_WORLD.chunkgen_i = i32vec3(0, 0, 0);

    VOXEL_WORLD.box.bound_min = VOXEL_CHUNKS[0].box.bound_min;
    VOXEL_WORLD.box.bound_max = VOXEL_CHUNKS[CHUNK_N - 1].box.bound_max;

    SCENE.sphere_n = 1;
    SCENE.spheres[0].o = f32vec3(-5, 0, 0);
    SCENE.spheres[0].r = 2;

    SCENE.box_n = 0;
}

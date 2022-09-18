#include <shared/shared.inl>

DAXA_USE_PUSH_CONSTANT(StartupCompPush)

#include <utils/voxel.glsl>

void startup_player() {
    PLAYER.pos = f32vec3(BLOCK_NX / VOXEL_SCL / 2, -BLOCK_NZ / VOXEL_SCL / 2, BLOCK_NZ / VOXEL_SCL / 2 * 3) + 0.001;
    PLAYER.vel = f32vec3(0, 0, 0);
    PLAYER.rot = f32vec3(-0.53, 0, 0.003);
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    startup_player();

    for (i32 zi = 0; zi < CHUNK_NZ; ++zi) {
        for (i32 yi = 0; yi < CHUNK_NY; ++yi) {
            for (i32 xi = 0; xi < CHUNK_NX; ++xi) {
                u32 index = get_chunk_index(i32vec3(xi, yi, zi));
                VOXEL_CHUNKS[index].box.bound_min = f32vec3(xi, yi, zi) * (CHUNK_SIZE / VOXEL_SCL);
                VOXEL_CHUNKS[index].box.bound_max = VOXEL_CHUNKS[index].box.bound_min + (CHUNK_SIZE / VOXEL_SCL);
            }
        }
    }

    VOXEL_WORLD.box.bound_min = VOXEL_CHUNKS[0].box.bound_min;
    VOXEL_WORLD.box.bound_max = VOXEL_CHUNKS[CHUNK_N - 1].box.bound_max;

    PLAYER.edit_radius = 1.0;

    SCENE.capsule_n++;
    SCENE.capsules[0].r = 0.3;
}

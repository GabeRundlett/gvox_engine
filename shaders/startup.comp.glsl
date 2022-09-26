#include <shared/shared.inl>

DAXA_USE_PUSH_CONSTANT(StartupCompPush)

#include <utils/voxel.glsl>

void startup_player() {
    PLAYER.pos = f32vec3(BLOCK_NX / VOXEL_SCL / 2, -BLOCK_NZ / VOXEL_SCL / 2, BLOCK_NZ / VOXEL_SCL / 2 * 3) + 0.001;
    PLAYER.vel = f32vec3(0, 0, 0);
    PLAYER.rot = f32vec3(-0.53, 0, 0.003);

    PLAYER.view_state = (PLAYER.view_state & ~(0x1 << 6)) | (1 << 6);
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    startup_player();

    for (i32 zi = 0; zi < CHUNK_NZ; ++zi) {
        for (i32 yi = 0; yi < CHUNK_NY; ++yi) {
            for (i32 xi = 0; xi < CHUNK_NX; ++xi) {
                u32 index = get_chunk_index(i32vec3(xi, yi, zi));
#if USING_BRICKMAP
#else
                VOXEL_CHUNKS[index].box.bound_min = f32vec3(xi, yi, zi) * (CHUNK_SIZE / VOXEL_SCL);
                VOXEL_CHUNKS[index].box.bound_max = VOXEL_CHUNKS[index].box.bound_min + (CHUNK_SIZE / VOXEL_SCL);
#endif
            }
        }
    }

#if USING_BRICKMAP
    // VOXEL_WORLD.chunk_is_ptr[0] = 0x1;

    VOXEL_WORLD.generation_chunk.box.bound_min = f32vec3(0, 0, 0) * (CHUNK_SIZE / VOXEL_SCL);
    VOXEL_WORLD.generation_chunk.box.bound_min = VOXEL_WORLD.generation_chunk.box.bound_min + (CHUNK_SIZE / VOXEL_SCL);

    VOXEL_WORLD.box.bound_min = f32vec3(0, 0, 0);
    VOXEL_WORLD.box.bound_max = f32vec3(CHUNK_NX, CHUNK_NY, CHUNK_NZ) * (CHUNK_SIZE / VOXEL_SCL);
#else
    VOXEL_WORLD.box.bound_min = VOXEL_CHUNKS[0].box.bound_min;
    VOXEL_WORLD.box.bound_max = VOXEL_CHUNKS[CHUNK_N - 1].box.bound_max * 1;
#endif

    PLAYER.edit_radius = 1.0;

    SCENE.capsule_n++;
    SCENE.capsules[0].r = 0.3;

    SCENE.box_n++;
    SCENE.boxes[0].bound_min = f32vec3(-10000, -10000, -1);
    SCENE.boxes[0].bound_max = f32vec3(+10000, +10000, +0);
}

#include <shared/shared.inl>

DAXA_USE_PUSH_CONSTANT(StartupCompPush)

#include <utils/math.glsl>
#include <utils/voxel.glsl>

void startup_player() {
    f32 p_offset = (BLOCK_NX + BLOCK_NZ) / 2 / VOXEL_SCL / 2;
    PLAYER.pos = f32vec3(-p_offset, -p_offset, BLOCK_NZ / VOXEL_SCL / 2 * 2.5) + 0.001;
    PLAYER.vel = f32vec3(0, 0, 0);
    PLAYER.rot = f32vec3(-0.53, 0, PI / 4);

    PLAYER.view_state = (PLAYER.view_state & ~(0x1 << 6)) | (1 << 6);
    PLAYER.edit_radius = 1.0;
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    startup_player();

    for (i32 zi = 0; zi < CHUNK_NZ; ++zi) {
        for (i32 yi = 0; yi < CHUNK_NY; ++yi) {
            for (i32 xi = 0; xi < CHUNK_NX; ++xi) {
                u32 index = get_chunk_index(i32vec3(xi, yi, zi));
#if USING_BRICKMAP
                VOXEL_WORLD.generation_chunk.box.bound_min = f32vec3(xi, yi, zi) * (CHUNK_SIZE / VOXEL_SCL);
                VOXEL_WORLD.generation_chunk.box.bound_min = VOXEL_WORLD.generation_chunk.box.bound_min + (CHUNK_SIZE / VOXEL_SCL);
#else
                VOXEL_CHUNKS[index].box.bound_min = f32vec3(xi, yi, zi) * (CHUNK_SIZE / VOXEL_SCL);
                VOXEL_CHUNKS[index].box.bound_max = VOXEL_CHUNKS[index].box.bound_min + (CHUNK_SIZE / VOXEL_SCL);
#endif
            }
        }
    }

#if USING_BRICKMAP
    // VOXEL_WORLD.chunk_is_ptr[0] = 0x1;
    VOXEL_WORLD.box.bound_min = f32vec3(0, 0, 0);
    VOXEL_WORLD.box.bound_max = f32vec3(CHUNK_NX, CHUNK_NY, CHUNK_NZ) * (CHUNK_SIZE / VOXEL_SCL);
#else
    VOXEL_WORLD.box.bound_min = VOXEL_CHUNKS[0].box.bound_min;
    VOXEL_WORLD.box.bound_max = VOXEL_CHUNKS[CHUNK_N - 1].box.bound_max * 1;
#endif

    SCENE.pick_box.bound_min = f32vec3(-1, -1, -1);
    SCENE.pick_box.bound_max = f32vec3(+1, +1, +1);

    SCENE.brush_origin_sphere.o = f32vec3(0, 0, 0);
    SCENE.brush_origin_sphere.r = 0.1;

    SCENE.capsules[SCENE.capsule_n].r = 0.05;
    SCENE.capsule_n++;

    // SCENE.boxes[SCENE.box_n].bound_min = f32vec3(-10000, -10000, -1);
    // SCENE.boxes[SCENE.box_n].bound_max = f32vec3(+10000, +10000, +0);
    // SCENE.box_n++;
}

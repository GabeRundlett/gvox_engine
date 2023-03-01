#include <shared/shared.inl>

DAXA_USE_PUSH_CONSTANT(StartupCompPush)

#include <utils/math.glsl>
#include <utils/voxel.glsl>

void startup_player() {
    f32 p_offset = (WORLD_BLOCK_NX + WORLD_BLOCK_NZ) / VOXEL_SCL / 64;
    PLAYER.pos = f32vec3(-p_offset, -p_offset, WORLD_BLOCK_NZ / VOXEL_SCL / 2) + 0.001;
    PLAYER.vel = f32vec3(0, 0, 0);
    PLAYER.rot = f32vec3(-0.53, 0, PI / 4);

    PLAYER.view_state = (PLAYER.view_state & ~(0x1 << 6)) | (1 << 6);
    PLAYER.edit_radius = 1.0;

    // Toggle brush on by default
    PLAYER.view_state |= 1 << 8;
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    startup_player();

    for (i32 zi = 0; zi < WORLD_CHUNK_NZ; ++zi) {
        for (i32 yi = 0; yi < WORLD_CHUNK_NY; ++yi) {
            for (i32 xi = 0; xi < WORLD_CHUNK_NX; ++xi) {
                u32 index = get_chunk_index_WORLD(i32vec3(xi, yi, zi));
                VOXEL_WORLD.voxel_chunks[index].box.bound_min = f32vec3(xi, yi, zi) * (CHUNK_SIZE / VOXEL_SCL);
                VOXEL_WORLD.voxel_chunks[index].box.bound_max = VOXEL_WORLD.voxel_chunks[index].box.bound_min + (CHUNK_SIZE / VOXEL_SCL);
            }
        }
    }

    for (i32 zi = 0; zi < BRUSH_CHUNK_NZ; ++zi) {
        for (i32 yi = 0; yi < BRUSH_CHUNK_NY; ++yi) {
            for (i32 xi = 0; xi < BRUSH_CHUNK_NX; ++xi) {
                u32 index = get_chunk_index_BRUSH(i32vec3(xi, yi, zi));
                VOXEL_BRUSH.voxel_chunks[index].box.bound_min = f32vec3(xi, yi, zi) * (CHUNK_SIZE / VOXEL_SCL);
                VOXEL_BRUSH.voxel_chunks[index].box.bound_max = VOXEL_WORLD.voxel_chunks[index].box.bound_min + (CHUNK_SIZE / VOXEL_SCL);
            }
        }
    }

    VOXEL_WORLD.box.bound_min = VOXEL_WORLD.voxel_chunks[0].box.bound_min;
    VOXEL_WORLD.box.bound_max = VOXEL_WORLD.voxel_chunks[WORLD_CHUNK_N - 1].box.bound_max;

    VOXEL_BRUSH.box.bound_min = f32vec3(-1, -1, -1);
    VOXEL_BRUSH.box.bound_max = f32vec3(+1, +1, +1);
    VOXEL_BRUSH.chunkgen_index = 0;

    SCENE.capsules[SCENE.capsule_n].r = 0.05;
    SCENE.capsule_n++;

    // SCENE.boxes[SCENE.box_n].bound_min = f32vec3(-10000, -10000, -1);
    // SCENE.boxes[SCENE.box_n].bound_max = f32vec3(+10000, +10000, +0);
    // SCENE.box_n++;
}

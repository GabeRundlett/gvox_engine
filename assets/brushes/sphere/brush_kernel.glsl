#pragma once

#if defined(CHUNKGEN)
#define OFFSET (f32vec3(WORLD_BLOCK_NX, WORLD_BLOCK_NY, WORLD_BLOCK_NZ) * 0.5 / VOXEL_SCL)
#else
#define OFFSET f32vec3(0, 0, 0)
#endif

void custom_brush_kernel(in BrushInput brush, inout Voxel result) {
    f32 l = length(brush.p - OFFSET) / PLAYER.edit_radius;
    f32 r = 1;
    b32 has_surrounding = true;

    // r = rand(brush.p + brush.origin + INPUT.time) - l;

    // f32vec3 voxel_p = brush.p + brush.origin;
    // Voxel v[6];
    // v[0] = unpack_voxel(sample_packed_voxel_WORLD(voxel_p + f32vec3(0, 0, -1) / VOXEL_SCL));
    // v[1] = unpack_voxel(sample_packed_voxel_WORLD(voxel_p + f32vec3(0, 0, +1) / VOXEL_SCL));
    // v[2] = unpack_voxel(sample_packed_voxel_WORLD(voxel_p + f32vec3(0, -1, 0) / VOXEL_SCL));
    // v[3] = unpack_voxel(sample_packed_voxel_WORLD(voxel_p + f32vec3(0, +1, 0) / VOXEL_SCL));
    // v[4] = unpack_voxel(sample_packed_voxel_WORLD(voxel_p + f32vec3(-1, 0, 0) / VOXEL_SCL));
    // v[5] = unpack_voxel(sample_packed_voxel_WORLD(voxel_p + f32vec3(+1, 0, 0) / VOXEL_SCL));
    // has_surrounding =
    //     v[0].block_id != BlockID_Air ||
    //     v[1].block_id != BlockID_Air ||
    //     v[2].block_id != BlockID_Air ||
    //     v[3].block_id != BlockID_Air ||
    //     v[4].block_id != BlockID_Air ||
    //     v[5].block_id != BlockID_Air;

    b32 should_break = INPUT.mouse.buttons[GAME_MOUSE_BUTTON_LEFT] != 0;

    if (l < 1 && r > 0.5 && has_surrounding) {
        result = Voxel(BRUSH_SETTINGS.color, BlockID_Stone);
    }
}

#pragma once

#if defined(CHUNKGEN)
#define OFFSET (f32vec3(BLOCK_NX, BLOCK_NY, BLOCK_NZ) * 0.5 / VOXEL_SCL)
#else
#define OFFSET f32vec3(0, 0, 0)
#endif

b32 custom_brush_should_edit(in BrushInput brush) {
    f32 r = 1; // rand(brush.p + brush.origin + INPUT.time);
    return length(brush.p - OFFSET) < PLAYER.edit_radius && r > 0.99;
}

Voxel custom_brush_kernel(in BrushInput brush) {
    return Voxel(INPUT.settings.brush_color, BlockID_Stone);
}

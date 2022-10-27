#pragma once

b32 custom_brush_should_edit(in BrushInput brush) {
    return length(brush.p) < PLAYER.edit_radius;
}

Voxel custom_brush_kernel(in BrushInput brush) {
    Voxel result = brush.prev_voxel;
    // if (brush.prev_voxel.block_id == BlockID_Stone)
    //     result.col = INPUT.settings.brush_color;
    f32 r = 1; // rand(brush.p + brush.origin + INPUT.time);
    if (brush.prev_voxel.block_id != BlockID_Air && r > 0.99)
        result.col = pow(INPUT.settings.brush_color, f32vec3(1.0));
    return result;
}

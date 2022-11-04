#pragma once

b32 custom_brush_should_edit(in BrushInput brush) {
    return length(brush.p) < PLAYER.edit_radius;
}

Voxel custom_brush_kernel(in BrushInput brush) {
    Voxel result = brush.prev_voxel;
    f32 r = rand(brush.p + brush.origin + INPUT.time);
    if (brush.prev_voxel.block_id != BlockID_Air && r > BRUSH_SETTINGS.random_threshold)
        result.col = pow(BRUSH_SETTINGS.color, f32vec3(2.2));
    return result;
}

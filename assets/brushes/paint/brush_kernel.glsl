#pragma once

void custom_brush_kernel(in BrushInput brush, inout Voxel result) {
    if (length(brush.p) < PLAYER.edit_radius) {
        result = brush.prev_voxel;
        f32 r = rand(brush.p + brush.origin + INPUT.time);
        if (brush.prev_voxel.block_id != BlockID_Air && r > BRUSH_SETTINGS.random_threshold)
            result.col = pow(BRUSH_SETTINGS.color, f32vec3(2.2));
    }
}

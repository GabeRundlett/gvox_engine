#pragma once

b32 custom_brush_should_edit(in BrushInput brush) {
    return length(brush.p) < PLAYER.edit_radius;
}

Voxel custom_brush_kernel(in BrushInput brush) {
    Voxel result = brush.prev_voxel;
    if (brush.prev_voxel.block_id == BlockID_Stone)
        result.col = INPUT.settings.brush_color;
    return result;
}

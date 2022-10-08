#pragma once

b32 custom_brush_should_edit(in BrushInput brush) {
    return length(brush.p) < PLAYER.edit_radius;
}

u32 custom_brush_id_kernel(in BrushInput brush) {
    return BlockID_Stone;
}

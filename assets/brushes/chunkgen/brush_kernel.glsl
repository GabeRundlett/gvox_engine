#pragma once

b32 custom_brush_should_edit(in BrushInput brush) {
    return true;
}

Voxel custom_brush_kernel(in BrushInput brush) {
    return gen_voxel(brush.p + brush.origin);
}

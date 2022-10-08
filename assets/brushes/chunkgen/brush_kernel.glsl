#pragma once

b32 custom_brush_should_edit(in BrushInput brush) {
    return true;
}

u32 custom_brush_id_kernel(in BrushInput brush) {
    Voxel voxel = gen_voxel(brush.p + brush.origin);
    return voxel.block_id;
}

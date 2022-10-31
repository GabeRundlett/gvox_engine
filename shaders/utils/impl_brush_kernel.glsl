#pragma once

#include <utils/impl_brush_header.glsl>
#include <brush_kernel.glsl>

b32 brush_should_edit(in f32vec3 voxel_p) {
    b32 result = false;
    BrushInput brush;
    brush.origin = floor((GLOBALS.brush_origin - GLOBALS.brush_offset) * VOXEL_SCL) / VOXEL_SCL;
    if (GLOBALS.edit_flags == 2) {
        brush.origin += GLOBALS.pick_intersection.nrm * 1.0 / VOXEL_SCL;
    }
    brush.p = voxel_p - brush.origin;
    brush.begin_p = GLOBALS.edit_origin - brush.origin;
    return (sd_box(voxel_p, VOXEL_BRUSH.box) < 0.0) && custom_brush_should_edit(brush);
}

Voxel brush_kernel(in f32vec3 voxel_p) {
    if (PLAYER.edit_voxel_id == BlockID_Air) {
        return Voxel(block_color(BlockID_Air), PLAYER.edit_voxel_id);
    } else {
        BrushInput brush;
        brush.origin = floor((GLOBALS.brush_origin - GLOBALS.brush_offset) * VOXEL_SCL) / VOXEL_SCL;
        brush.p = voxel_p - brush.origin;
        brush.begin_p = GLOBALS.edit_origin - brush.origin;
        brush.prev_voxel = unpack_voxel(sample_packed_voxel(voxel_p));
        return custom_brush_kernel(brush);
    }
}

#pragma once

#include <brush_input.glsl>
#include <utils/impl_brush_header.glsl>
#include <brush_kernel.glsl>

Voxel brush_kernel(in f32vec3 voxel_p) {
    Voxel result;
    result.block_id = BlockID_Debug;
    if (sd_box(voxel_p, VOXEL_BRUSH.box) <= 0.0) {
        BrushInput brush;
        brush.origin = GLOBALS.brush_origin;
        brush.p = voxel_p - brush.origin;
        brush.begin_p = GLOBALS.edit_origin - brush.origin;
        brush.prev_voxel = unpack_voxel(sample_packed_voxel_WORLD(voxel_p));
        custom_brush_kernel(brush, result);
        if (PLAYER.edit_voxel_id == BlockID_Air) {
            if (result.block_id != BlockID_Debug)
                result = Voxel(block_color(BlockID_Air), PLAYER.edit_voxel_id);
        }
    }
    return result;
}

#pragma once

void custom_brush_kernel(in BrushInput brush, inout Voxel result) {
    PackedVoxel packed_voxel = sample_gvox_palette_voxel(brush.p);
    if (packed_voxel.data != 0) {
        result = unpack_voxel(packed_voxel);
        result.block_id = BlockID_Stone;
    }
}

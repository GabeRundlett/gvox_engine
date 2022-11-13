#pragma once

BoundingBox custom_brush_box() {
    return BoundingBox(
        f32vec3(-2, -2, -0.15 - 1.0 / VOXEL_SCL),
        f32vec3(+2, +2, +5.15 + 1.0 / VOXEL_SCL));
}

f32vec3 custom_brush_origin_offset() {
    return f32vec3(0.0, 0.0, 4.0);
}

b32 custom_brush_enable(f32vec3 p0, f32vec3 p1) {
    u32 block_id = sample_voxel_id_WORLD(p1);
    return block_id == BlockID_Grass || block_id == BlockID_TallGrass;
}

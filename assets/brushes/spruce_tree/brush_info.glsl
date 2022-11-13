#pragma once

f32vec3 custom_brush_size() {
    return f32vec3(+4, +4, +5.5);
}

f32vec3 custom_brush_origin_offset() {
    return f32vec3(-2, -2, -1.0 / VOXEL_SCL);
}

b32 custom_brush_enable(f32vec3 p0, f32vec3 p1) {
    u32 block_id = sample_voxel_id_WORLD(p1);
    return block_id == BlockID_Grass || block_id == BlockID_TallGrass;
}

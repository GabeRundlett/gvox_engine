#pragma once

Box custom_brush_box() {
    return Box(
        f32vec3(-8, -8, -0.15 - 1.0 / VOXEL_SCL),
        f32vec3(+8, +8, +20.15 + 1.0 / VOXEL_SCL));
}

f32vec3 custom_brush_origin_offset() {
    return f32vec3(0.0, 0.0, 16.0);
}

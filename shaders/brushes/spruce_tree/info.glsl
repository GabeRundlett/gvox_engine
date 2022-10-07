#pragma once

f32vec3 custom_brush_range() {
    return f32vec3(4.0, 4.0, 5.3 + 2.0 / VOXEL_SCL);
}

f32vec3 custom_brush_origin_offset() {
    return f32vec3(0.0, 0.0, 10.15);
}

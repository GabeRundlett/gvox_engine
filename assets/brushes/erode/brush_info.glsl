#pragma once

BoundingBox custom_brush_box() {
    return BoundingBox(
        f32vec3(-PLAYER.edit_radius),
        f32vec3(+PLAYER.edit_radius));
}

f32vec3 custom_brush_origin_offset() {
    return f32vec3(0.0, 0.0, 0.0);
}

b32 custom_brush_enable(f32vec3 p0, f32vec3 p1) {
    return true;
}

#pragma once

f32vec3 custom_brush_size() {
    return f32vec3(2.0 * PLAYER.edit_radius);
}

f32vec3 custom_brush_origin_offset() {
    return f32vec3(-1.0 * PLAYER.edit_radius, -1.0 * PLAYER.edit_radius, 0);
}

b32 custom_brush_enable(f32vec3 p0, f32vec3 p1) {
    return true;
}

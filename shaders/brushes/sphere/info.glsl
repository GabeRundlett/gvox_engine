#pragma once

f32vec3 custom_brush_range() {
    return f32vec3(PLAYER.edit_radius * 2.0);
}

f32vec3 custom_brush_origin_offset() {
    return f32vec3(0, 0, 0);
}

#pragma once

Box custom_brush_box() {
    return Box(
        f32vec3(-PLAYER.edit_radius),
        f32vec3(+PLAYER.edit_radius));
}

f32vec3 custom_brush_origin_offset() {
    return f32vec3(0.0, 0.0, 0.0);
}

#pragma once

#include <utils/voxel.glsl>
#include <utils/noise.glsl>
#include <utils/sd_shapes.glsl>

f32 brush_noise_value(in f32vec3 voxel_p) {
    FractalNoiseConfig noise_conf = FractalNoiseConfig(
        /* .amplitude   = */ 1.0,
        /* .persistance = */ 0.5,
        /* .scale       = */ 0.5,
        /* .lacunarity  = */ 2.0,
        /* .octaves     = */ 4);
    return fractal_noise(voxel_p, noise_conf);
}

f32 sd_gear(vec3 p, float radius1, float radius2, f32 notch_n, f32 notch_fac) {
    vec2 q = p.xy;
    float y = length(p.xy) - 0.2;

    radius2 = radius1 - radius2;
    f32 n = 1.0 / notch_n;
    f32 open_factor = 0.5;

    const float an = 6.283185 * n;
    float fa = (atan(q.y, q.x) + an * 0.5) / an;
    float sym = an * floor(fa);
    vec2 r = mat2(cos(sym), -sin(sym), sin(sym), cos(sym)) * q;
    f32 d = radius1 * length(max(abs(r - vec2(radius1, 0)) - vec2(radius2 * 0.5, y * n * 4.0 * notch_fac), 0.0));
    d = d - 0.001;
    d = max(d, -sd_plane(p + f32vec3(0, 0, 1)));
    d = max(d, +sd_plane(p - f32vec3(0, 0, 1)));

    return d;
}

f32 sd_castle_wall(in f32vec3 p) {
    f32 val = MAX_SD;

    // t_* = tower
    f32 t_repeat_x = 30;

    f32vec3 t_p = p;
    t_p.x = t_p.x = (fract(t_p.x / t_repeat_x + 0.5) - 0.5) * t_repeat_x;

    val = min(val, sd_box(t_p - f32vec3(0.0, 0.0, 0.0), f32vec3(t_repeat_x, 1, 4)));
    val = min(val, sd_box(t_p - f32vec3(0.0, +1.0, 4.0), f32vec3(t_repeat_x, 0.1, 1)));
    val = min(val, sd_box(t_p - f32vec3(0.0, -1.0, 4.0), f32vec3(t_repeat_x, 0.1, 1)));

    // wub_* = wall upper battlements
    f32 wub_repeat_x = 1.2;
    f32vec3 wub_p = p;
    wub_p.x = wub_p.x = (fract(wub_p.x / wub_repeat_x + 0.5) - 0.5) * wub_repeat_x;
    val = min(val, sd_box(wub_p - f32vec3(0.0, +1.0, 5.0), f32vec3(0.45, 0.1, 0.4)));
    val = min(val, sd_box(wub_p - f32vec3(0.0, -1.0, 5.0), f32vec3(0.45, 0.1, 0.4)));

    // wlb_* = wall lower battlements
    f32 wlb_repeat_x = 4;
    f32vec3 wlb_p = p;
    wlb_p.x = wlb_p.x = (fract(wlb_p.x / wlb_repeat_x + 0.5) - 0.5) * wlb_repeat_x;
    wlb_p.z -= 2;
    val = min(val, sd_triangular_prism(wlb_p, 1.85, 1));
    val = min(val, sd_box(wlb_p + f32vec3(0, 0, 2.8), f32vec3(1, 1.6, 2)));

    val = min(val, sd_cylinder(t_p, 4, 8));
    val = max(val, -sd_cylinder(t_p, 3.2, 8));

    val = min(val, sd_gear(t_p - f32vec3(0, 0, 8), 4, 3.2, 12, 0.5));

    val = max(val, -sd_plane(p - f32vec3(0, 0, -4)));
    return val;
}

b32 brush_should_edit(in f32vec3 voxel_p) {
    voxel_p = voxel_p;
    b32 result = false;
    f32vec3 pick_pos = floor(GLOBALS.pick_pos * VOXEL_SCL) / VOXEL_SCL;
    f32vec3 p = voxel_p - GLOBALS.edit_origin;
    result = sd_castle_wall(p) + abs(brush_noise_value(voxel_p)) * 0.0 < 0.0;
    return result && (sd_box(voxel_p - pick_pos, f32vec3(8, 8, 8)) < 0.0);
}
u32 brush_id_kernel(in f32vec3 voxel_p) {
    return PLAYER.edit_voxel_id;
}

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
    f32 d = length(max(abs(r - vec2(radius1, 0)) - vec2(radius2 * 0.5, y * n * 4.0 * notch_fac), 0.0));
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

    val = min(val, sd_box(t_p, f32vec3(t_repeat_x, 1, 4)));

    // wlb_* = wall lower battlements
    f32 wlb_repeat_x = 4;
    f32vec3 wlb_p = p;
    wlb_p.x = wlb_p.x = (fract(wlb_p.x / wlb_repeat_x + 0.5) - 0.5) * wlb_repeat_x;
    wlb_p.z -= 2;
    val = min(val, sd_triangular_prism(wlb_p, 1.85, 1));
    val = min(val, sd_box(wlb_p + f32vec3(0, 0, 2.8), f32vec3(1, 1.6, 2)));

    // wub_* = wall upper battlements

    val = min(val, sd_cylinder(t_p, 4, 8));
    val = max(val, -sd_cylinder(t_p, 3.2, 8));

    val = min(val, sd_gear(t_p - f32vec3(0, 0, 8), 4, 3.2, 12, 0.5));

    val = max(val, -sd_plane(p - f32vec3(0, 0, -4)));
    return val;
    // f32vec3 p = abs(voxel_p - GLOBALS.pick_pos) - PLAYER.edit_radius;
    // f32 wall_x = voxel_p.y - GLOBALS.edit_origin.y;
    // f32 val = 0;
    // val += f32(p.z > 1.0);
    // val += f32(wall_x > +2.0 || wall_x < -2.0);
    // val += min(f32(fract(voxel_p.x) > 0.5) + f32(wall_x < +1.9 && wall_x > -1.9), f32(voxel_p.z > 23.75));
    // val += min(f32(wall_x < +1.9 && wall_x > -1.9), f32(voxel_p.z > 22.5));
    // val += f32(voxel_p.z > 24.0);
    // val -= f32(max(p.x, p.x) < 1.0);
    // val = -200 * f32(val < 0.0) + 0.5;
    // f32vec2 c_uv = voxel_p.xy - f32vec2(round(GLOBALS.pick_pos.x / 24) * 24, GLOBALS.edit_origin.y);
    // f32 c_r = length(c_uv);
    // f32 angle = (atan(c_uv.y, c_uv.x) / 3.14159 * 0.5) + 0.5;
    // val += f32(voxel_p.z > 30.0);
    // val += f32((fract(angle * 20 - 0.25) < 0.5) && voxel_p.z > 29.75);
    // val -= f32(c_r < 4 && (c_r > 3.8 || (voxel_p.z < 28.75 && voxel_p.z > 28.5)) && (wall_x > +0.5 || wall_x < -0.5 || voxel_p.z > 24.7));
    // f32 stair_angle = angle * 45 - 0.25;
    // f32 stair_height = floor(stair_angle) * 0.2 / 3;
    // val -= f32(voxel_p.z < 28.75 && c_r < 4 && c_r > 2.5 && (fract(stair_angle) < 0.75) && fract(voxel_p.z / 4) * 4 > 0.75 + stair_height && fract(voxel_p.z / 4) * 4 < 1 + stair_height);
    // val = -200 * f32(val < 0.0) + 0.5;
    // val += f32(voxel_p.z < 4.0) * 10000;
    // return val;
}

b32 brush_should_edit(in f32vec3 voxel_p) {
    voxel_p = voxel_p;
    b32 result = false;
    f32vec3 pick_pos = floor(GLOBALS.pick_pos * VOXEL_SCL) / VOXEL_SCL;
    f32vec3 p = voxel_p - GLOBALS.edit_origin;
    // result = sd_sphere(voxel_p - pick_pos, PLAYER.edit_radius) < 0.0;
    // result = sd_cylinder(p, PLAYER.edit_radius, PLAYER.edit_radius) < 0.0;
    // result = sd_sphere(voxel_p - pick_pos, PLAYER.edit_radius) + brush_noise_value(voxel_p) * 5.0 < 0.0;
    result = sd_castle_wall(p) + (brush_noise_value(voxel_p) + 1.01) * 0.0 < 0.0;
    return result && (sd_box(voxel_p - pick_pos, f32vec3(8, 8, 8)) < 0.0);
}
u32 brush_id_kernel(in f32vec3 voxel_p) {
    return PLAYER.edit_voxel_id;
}

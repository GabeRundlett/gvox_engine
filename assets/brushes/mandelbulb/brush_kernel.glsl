#pragma once

b32 mandelbrot(in f32vec3 c) {
    f32vec2 z = c.xy;
    u32 i = 0;
    for (; i < 100; ++i) {
        f32vec2 z_ = z;
        z.x = z_.x * z_.x - z_.y * z_.y;
        z.y = 2.0 * z_.x * z_.y;
        z += c.xy;
        if (dot(z, z) > 4)
            break;
    }
    return i == 100;
}

b32 mandelbulb(in f32vec3 c) {
    f32vec3 z = c;
    u32 i = 0;
    const f32 n = 8;
    const u32 MAX_ITER = 20;
    for (; i < MAX_ITER; ++i) {
        f32 r = length(z);
        f32 p = atan(z.y / z.x);
        f32 t = acos(z.z / r);
        z = f32vec3(
            sin(n * t) * cos(n * p),
            sin(n * t) * sin(n * p),
            cos(n * t));
        z = z * pow(r, n) + c;
        if (dot(z, z) > 4)
            break;
    }
    return i == MAX_ITER;
}

#if defined(CHUNKGEN)
#define OFFSET (f32vec3(BLOCK_NX, BLOCK_NY, BLOCK_NZ) * 0.5 / VOXEL_SCL)
#define BRUSH_SCL (min(BLOCK_NX, min(BLOCK_NY, BLOCK_NZ)) * 0.5 / VOXEL_SCL)
#else
#define OFFSET f32vec3(0, 0, 0)
#define BRUSH_SCL PLAYER.edit_radius
#endif

b32 custom_brush_should_edit(in BrushInput brush) {
    f32vec3 a = brush.p;
    a = a - OFFSET;
    f32vec3 c = a * 1.0 / BRUSH_SCL;
    return mandelbulb(c);
}

Voxel custom_brush_kernel(in BrushInput brush) {
    return Voxel(INPUT.settings.brush_color, BlockID_Stone);
}

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

b32 mandelbulb(in f32vec3 c, in out f32vec3 color) {
    f32vec3 z = c;
    u32 i = 0;
    const f32 n = 8;
    const u32 MAX_ITER = 4;
    f32 m = dot(z, z);
    f32vec4 trap = f32vec4(abs(z), m);
    for (; i < MAX_ITER; ++i) {
        f32 r = length(z);
        f32 p = atan(z.y / z.x);
        f32 t = acos(z.z / r);
        z = f32vec3(
            sin(n * t) * cos(n * p),
            sin(n * t) * sin(n * p),
            cos(n * t));
        z = z * pow(r, n) + c;
        trap = min(trap, f32vec4(abs(z), m));
        m = dot(z, z);
        if (m > 256.0)
            break;
    }
    color = f32vec3(m, trap.yz) * trap.w;
    return i == MAX_ITER;
}

#if defined(CHUNKGEN)
#define OFFSET (f32vec3(WORLD_BLOCK_NX, WORLD_BLOCK_NY, WORLD_BLOCK_NZ) * 0.5 / VOXEL_SCL)
#define BRUSH_SCL (min(WORLD_BLOCK_NX, min(WORLD_BLOCK_NY, WORLD_BLOCK_NZ)) * 0.5 / VOXEL_SCL)
#else
#define OFFSET f32vec3(0, 0, 0)
#define BRUSH_SCL PLAYER.edit_radius
#endif

void custom_brush_kernel(in BrushInput brush, inout Voxel result) {
    f32vec3 a = brush.p;
    a = a - OFFSET;
    f32vec3 c = a * 1.2 / BRUSH_SCL;
    f32vec3 color;
    if (mandelbulb(c, color)) {
        result = Voxel(mix(1 - BRUSH_SETTINGS.color, BRUSH_SETTINGS.color, clamp(color.r, 0, 1)), BlockID_Stone);
    }
}

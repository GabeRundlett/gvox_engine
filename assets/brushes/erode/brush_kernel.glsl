#pragma once

#if defined(CHUNKGEN)
#define OFFSET (f32vec3(WORLD_BLOCK_NX, WORLD_BLOCK_NY, WORLD_BLOCK_NZ) * 0.5 / VOXEL_SCL)
#else
#define OFFSET f32vec3(0, 0, 0)
#endif

f32 erode_noise(f32vec3 p) {
    FractalNoiseConfig noise_conf = FractalNoiseConfig(
        /* .amplitude   = */ 1.0,
        /* .persistance = */ 0.5,
        /* .scale       = */ 0.1,
        /* .lacunarity  = */ 4.0,
        /* .octaves     = */ 4);
    f32 val = fractal_noise(p, noise_conf);
    return val;
}

void custom_brush_kernel(in BrushInput brush, inout Voxel result) {
    f32 random_factor = rand(brush.origin);
    f32 n = erode_noise(brush.p + brush.origin + INPUT.time * 100);
    f32 l = length(brush.p - OFFSET) / (PLAYER.edit_radius + random_factor);
    if (n > pow(l, 2) * 0.5 - 0.25) {
        f32vec3 voxel_p = brush.p + brush.origin;
        Voxel v[6];
        v[0 + 0] = unpack_voxel(sample_packed_voxel_WORLD(voxel_p + f32vec3(0, 0, -1) / VOXEL_SCL));
        v[1 + 0] = unpack_voxel(sample_packed_voxel_WORLD(voxel_p + f32vec3(0, 0, +1) / VOXEL_SCL));
        v[2 + 0] = unpack_voxel(sample_packed_voxel_WORLD(voxel_p + f32vec3(0, -1, 0) / VOXEL_SCL));
        v[3 + 0] = unpack_voxel(sample_packed_voxel_WORLD(voxel_p + f32vec3(0, +1, 0) / VOXEL_SCL));
        v[4 + 0] = unpack_voxel(sample_packed_voxel_WORLD(voxel_p + f32vec3(-1, 0, 0) / VOXEL_SCL));
        v[5 + 0] = unpack_voxel(sample_packed_voxel_WORLD(voxel_p + f32vec3(+1, 0, 0) / VOXEL_SCL));
        u32 air_count = 0;
        u32 non_air_id = 0;
        for (u32 i = 0; i < 6; ++i) {
            if (v[i].block_id == BlockID_Air)
                air_count += 1;
            else {
                non_air_id = i;
            }
        }
        if (air_count > 0) {
            result = Voxel(block_color(BlockID_Air), BlockID_Air);
        }
    }

    // f32 l = length(brush.p - OFFSET) / (PLAYER.edit_radius + random_factor);
    // if (l < 1 && r > pow(l, 16) * 0.5) {
    // }
}

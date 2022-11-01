#pragma once

#if defined(CHUNKGEN)
#define OFFSET (f32vec3(WORLD_BLOCK_NX, WORLD_BLOCK_NY, WORLD_BLOCK_NZ) * 0.5 / VOXEL_SCL)
#else
#define OFFSET f32vec3(0, 0, 0)
#endif

b32 custom_brush_should_edit(in BrushInput brush) {
    f32 l = length(brush.p - OFFSET) / PLAYER.edit_radius;
    return l < 1;
}

Voxel custom_brush_kernel(in BrushInput brush) {
    f32vec3 voxel_p = brush.p + brush.origin;
    Voxel v[6];
    v[0 + 0] = unpack_voxel(sample_packed_voxel(voxel_p + f32vec3(0, 0, -1) / VOXEL_SCL));
    v[1 + 0] = unpack_voxel(sample_packed_voxel(voxel_p + f32vec3(0, 0, +1) / VOXEL_SCL));
    v[2 + 0] = unpack_voxel(sample_packed_voxel(voxel_p + f32vec3(0, -1, 0) / VOXEL_SCL));
    v[3 + 0] = unpack_voxel(sample_packed_voxel(voxel_p + f32vec3(0, +1, 0) / VOXEL_SCL));
    v[4 + 0] = unpack_voxel(sample_packed_voxel(voxel_p + f32vec3(-1, 0, 0) / VOXEL_SCL));
    v[5 + 0] = unpack_voxel(sample_packed_voxel(voxel_p + f32vec3(+1, 0, 0) / VOXEL_SCL));

    u32 air_count = 0;
    u32 non_air_id = 0;

    for (u32 i = 0; i < 6; ++i) {
        if (v[i].block_id == BlockID_Air)
            air_count += 1;
        else {
            non_air_id = i;
        }
    }

    if (air_count < 1) {
        return v[non_air_id];
    } else {
        return Voxel(block_color(BlockID_Air), BlockID_Air);
    }
}

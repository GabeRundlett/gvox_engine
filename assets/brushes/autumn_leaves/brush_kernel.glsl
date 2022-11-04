#pragma once

#if defined(CHUNKGEN)
#define OFFSET (f32vec3(WORLD_BLOCK_NX, WORLD_BLOCK_NY, WORLD_BLOCK_NZ) * 0.5 / VOXEL_SCL)
#else
#define OFFSET f32vec3(0, 0, 0)
#endif

b32 custom_brush_should_edit(in BrushInput brush) {
    f32 l = length(brush.p - OFFSET) / PLAYER.edit_radius;
    f32 r = 1;
    b32 has_surrounding = true;

    r = rand(brush.p + brush.origin + INPUT.time) - l * 0.1;

    f32vec3 voxel_p = brush.p + brush.origin;
    Voxel v[6];
    v[0] = unpack_voxel(sample_packed_voxel(voxel_p + f32vec3(0, 0, -1) / VOXEL_SCL));
    v[1] = unpack_voxel(sample_packed_voxel(voxel_p + f32vec3(0, 0, +1) / VOXEL_SCL));
    v[2] = unpack_voxel(sample_packed_voxel(voxel_p + f32vec3(0, -1, 0) / VOXEL_SCL));
    v[3] = unpack_voxel(sample_packed_voxel(voxel_p + f32vec3(0, +1, 0) / VOXEL_SCL));
    v[4] = unpack_voxel(sample_packed_voxel(voxel_p + f32vec3(-1, 0, 0) / VOXEL_SCL));
    v[5] = unpack_voxel(sample_packed_voxel(voxel_p + f32vec3(+1, 0, 0) / VOXEL_SCL));
    has_surrounding =
        v[0].block_id == BlockID_Grass || v[0].block_id == BlockID_Leaves ||
        v[1].block_id == BlockID_Grass || v[1].block_id == BlockID_Leaves ||
        v[2].block_id == BlockID_Grass || v[2].block_id == BlockID_Leaves ||
        v[3].block_id == BlockID_Grass || v[3].block_id == BlockID_Leaves ||
        v[4].block_id == BlockID_Grass || v[4].block_id == BlockID_Leaves ||
        v[5].block_id == BlockID_Grass || v[5].block_id == BlockID_Leaves;

    return l < 1 && r > 0.95 && has_surrounding && unpack_voxel(sample_packed_voxel(voxel_p)).block_id == BlockID_Air;
}

Voxel custom_brush_kernel(in BrushInput brush) {
    return Voxel(pow(BRUSH_SETTINGS.color, f32vec3(2.2)), BlockID_Leaves);
}

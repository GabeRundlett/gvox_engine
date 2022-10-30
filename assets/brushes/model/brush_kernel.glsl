#pragma once

#if defined(CHUNKGEN)
#define OFFSET f32vec3(0, 0, 0)
#else
#define OFFSET brush.begin_p
#endif

b32 custom_brush_should_edit(in BrushInput brush) {
    i32vec3 p = i32vec3((brush.p - OFFSET) * VOXEL_SCL);
    if (p.x < 0 || p.y < 0 || p.z < 0 ||
        p.x >= MODEL.size_x || p.y >= MODEL.size_y || p.z >= MODEL.size_z) {
        return false;
    }
    u32 i = p.x + p.y * MODEL.size_x + p.z * MODEL.size_x * MODEL.size_y;
    GVoxModelVoxel vox = MODEL.voxels[i];

    return vox.id != 0;
}

Voxel custom_brush_kernel(in BrushInput brush) {
    i32vec3 p = i32vec3((brush.p - OFFSET) * VOXEL_SCL);
    if (p.x < 0 || p.y < 0 || p.z < 0 ||
        p.x >= MODEL.size_x || p.y >= MODEL.size_y || p.z >= MODEL.size_z) {
        return Voxel(block_color(BlockID_Stone), BlockID_Stone);
    }
    u32 i = p.x + p.y * MODEL.size_x + p.z * MODEL.size_x * MODEL.size_y;
    GVoxModelVoxel vox = MODEL.voxels[i];

    return Voxel(pow(vox.col, f32vec3(2.2)), BlockID_Stone);
}

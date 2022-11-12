#pragma once

#if defined(CHUNKGEN)
#define OFFSET f32vec3(0, 0, 0)
#else
#define OFFSET brush.begin_p
#endif

void custom_brush_kernel(in BrushInput brush, inout Voxel result) {
    b32 should_edit = true;

    i32vec3 p = i32vec3((brush.p - OFFSET) * VOXEL_SCL);
    if (p.x < 0 || p.y < 0 || p.z < 0 ||
        p.x >= MODEL.size_x || p.y >= MODEL.size_y || p.z >= MODEL.size_z) {
        return;
    }
    u32 i = p.x + p.y * MODEL.size_x + p.z * MODEL.size_x * MODEL.size_y;
    GVoxModelVoxel vox = MODEL.voxels[i];

    if (vox.id != 0) {
        result = Voxel(pow(vox.col, f32vec3(2.2)), BlockID_Stone);
    }
}

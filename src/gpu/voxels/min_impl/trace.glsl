#pragma once

#include <shared/app.inl>

VoxelTraceResult voxel_trace(in VoxelTraceInfo info, in out f32vec3 ray_pos) {
    VoxelTraceResult result;

    result.dist = 0;
    result.nrm = -info.ray_dir;
    result.step_n = 0;
    result.voxel_data = 0;

    if (info.extend_to_max_dist) {
        result.dist = MAX_DIST;
    }

    return result;
}

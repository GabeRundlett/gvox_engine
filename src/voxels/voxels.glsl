#pragma once

#include <voxels/voxels.inl>

struct VoxelTraceInfo {
    VoxelBufferPtrs ptrs;
    vec3 ray_dir;
    uint max_steps;
    float max_dist;
    float angular_coverage;
    bool extend_to_max_dist;
};

struct VoxelTraceResult {
    float dist;
    vec3 nrm;
    vec3 vel;
    uint step_n;
    PackedVoxel voxel_data;
};

// These are the functions that this file must define!
VoxelTraceResult voxel_trace(in VoxelTraceInfo info, in out vec3 ray_pos);

#include <voxels/impl/trace.glsl>

#pragma once

#include <shared/app.inl>

struct VoxelTraceInfo {
    VoxelTraceInfoPtrs ptrs;
    u32vec3 chunk_n;
    f32vec3 ray_dir;
    u32 max_steps;
    f32 max_dist;
    f32 angular_coverage;
    bool extend_to_max_dist;
};

struct VoxelTraceResult {
    f32 dist;
    f32vec3 nrm;
    u32 step_n;
    u32 voxel_data;
};

// These are the functions that this file must define!
VoxelTraceResult voxel_trace(in VoxelTraceInfo info, in out f32vec3 ray_pos);

#include <voxels/impl/trace.glsl>
#include <voxels/impl/voxel_world.glsl>

#if !defined(VOXEL_TRACE_INFO_PTRS)
#error "The implementation must define a way for the users to construct the trace info pointers!"
#endif

#if !defined(VOXEL_PERFRAME_INFO_PTRS)
#error "The implementation must define a way for the users to construct the perframe info pointers!"
#endif

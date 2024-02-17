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

void voxel_world_startup(daxa_RWBufferPtr(GpuGlobals) globals_ptr, VoxelRWBufferPtrs ptrs);
void voxel_world_perframe(daxa_BufferPtr(GpuInput) gpu_input, daxa_RWBufferPtr(GpuOutput) gpu_output, daxa_RWBufferPtr(GpuGlobals) globals_ptr, VoxelRWBufferPtrs ptrs);

#include <voxels/impl/trace.glsl>
#include <voxels/impl/voxel_world.glsl>

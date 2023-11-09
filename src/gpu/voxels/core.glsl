#pragma once

#include <shared/voxels/voxels.inl>

struct VoxelTraceInfo {
    VoxelBufferPtrs ptrs;
    f32vec3 ray_dir;
    u32 max_steps;
    f32 max_dist;
    f32 angular_coverage;
    bool extend_to_max_dist;
};

struct VoxelTraceResult {
    f32 dist;
    f32vec3 nrm;
    f32vec3 vel;
    u32 step_n;
    u32 voxel_data;
};

// These are the functions that this file must define!
VoxelTraceResult voxel_trace(in VoxelTraceInfo info, in out f32vec3 ray_pos);

void voxel_world_startup(daxa_RWBufferPtr(GpuGlobals) globals_ptr, VoxelRWBufferPtrs ptrs);
void voxel_world_perframe(daxa_BufferPtr(GpuInput) gpu_input, daxa_RWBufferPtr(GpuOutput) gpu_output, daxa_RWBufferPtr(GpuGlobals) globals_ptr, VoxelRWBufferPtrs ptrs);

#include <voxels/impl/trace.glsl>
#include <voxels/impl/voxel_world.glsl>

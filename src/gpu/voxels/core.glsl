#pragma once

#include <shared/voxels/voxels.inl>

struct VoxelTraceInfo {
    VoxelBufferPtrs ptrs;
    daxa_f32vec3 ray_dir;
    daxa_u32 max_steps;
    daxa_f32 max_dist;
    daxa_f32 angular_coverage;
    bool extend_to_max_dist;
};

struct VoxelTraceResult {
    daxa_f32 dist;
    daxa_f32vec3 nrm;
    daxa_f32vec3 vel;
    daxa_u32 step_n;
    daxa_u32 voxel_data;
};

// These are the functions that this file must define!
VoxelTraceResult voxel_trace(in VoxelTraceInfo info, in out daxa_f32vec3 ray_pos);

void voxel_world_startup(daxa_RWBufferPtr(GpuGlobals) globals_ptr, VoxelRWBufferPtrs ptrs);
void voxel_world_perframe(daxa_BufferPtr(GpuInput) gpu_input, daxa_RWBufferPtr(GpuOutput) gpu_output, daxa_RWBufferPtr(GpuGlobals) globals_ptr, VoxelRWBufferPtrs ptrs);

#include <voxels/impl/trace.glsl>
#include <voxels/impl/voxel_world.glsl>

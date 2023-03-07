#pragma once

#include <shared/settings.inl>
#include <shared/input.inl>
#include <shared/gvox_model.inl>
#include <shared/globals.inl>
#include <shared/voxels.inl>

#include <shared/core.inl>

struct StartupComputePush {
    daxa_BufferPtr(GpuSettings) gpu_settings;
    daxa_RWBufferPtr(GpuGlobals) gpu_globals;
    daxa_RWBufferPtr(VoxelChunk) voxel_chunks;
};

struct PerframeComputePush {
    daxa_BufferPtr(GpuSettings) gpu_settings;
    daxa_BufferPtr(GpuInput) gpu_input;
    daxa_RWBufferPtr(GpuGlobals) gpu_globals;
    daxa_RWBufferPtr(VoxelChunk) voxel_chunks;
};

struct PerChunkComputePush {
    daxa_BufferPtr(GpuSettings) gpu_settings;
    daxa_BufferPtr(GpuInput) gpu_input;
    daxa_RWBufferPtr(GpuGlobals) gpu_globals;
    daxa_RWBufferPtr(VoxelChunk) voxel_chunks;
};

struct ChunkOptComputePush {
    daxa_BufferPtr(GpuSettings) gpu_settings;
    daxa_BufferPtr(GpuInput) gpu_input;
    daxa_BufferPtr(GpuGlobals) gpu_globals;
    daxa_BufferPtr(GpuGvoxModel) gpu_gvox_model;
    daxa_RWBufferPtr(VoxelChunk) voxel_chunks;
};

struct TracePrimaryComputePush {
    daxa_BufferPtr(GpuSettings) gpu_settings;
    daxa_BufferPtr(GpuInput) gpu_input;
    daxa_BufferPtr(GpuGlobals) gpu_globals;
    daxa_BufferPtr(GpuGvoxModel) gpu_gvox_model;
    daxa_BufferPtr(VoxelChunk) voxel_chunks;
    daxa_RWImage2Df32 render_pos_image_id;
};

struct ColorSceneComputePush {
    daxa_BufferPtr(GpuSettings) gpu_settings;
    daxa_BufferPtr(GpuInput) gpu_input;
    daxa_BufferPtr(GpuGlobals) gpu_globals;
    daxa_BufferPtr(GpuGvoxModel) gpu_gvox_model;
    daxa_BufferPtr(VoxelChunk) voxel_chunks;
    daxa_RWImage2Df32 render_pos_image_id;
    daxa_RWImage2Df32 render_col_image_id;
};

struct PostprocessingComputePush {
    daxa_BufferPtr(GpuSettings) gpu_settings;
    daxa_BufferPtr(GpuInput) gpu_input;
    daxa_BufferPtr(GpuGlobals) gpu_globals;
    daxa_BufferPtr(GpuGvoxModel) gpu_gvox_model;
    daxa_BufferPtr(VoxelChunk) voxel_chunks;
    daxa_RWImage2Df32 render_col_image_id;
    daxa_RWImage2Df32 final_image_id;
};

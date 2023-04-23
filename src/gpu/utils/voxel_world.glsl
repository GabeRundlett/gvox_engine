#pragma once

#include <shared/shared.inl>

#include <utils/math.glsl>
#include <utils/voxels.glsl>

#define PLAYER deref(globals_ptr).player
#define VOXEL_WORLD deref(globals_ptr).voxel_world
#define CHUNKS(i) deref(voxel_chunks_ptr + i)
void voxel_world_startup(
    daxa_RWBufferPtr(GpuGlobals) globals_ptr,
    daxa_RWBufferPtr(VoxelChunk) voxel_chunks_ptr) {

    VOXEL_WORLD.chunk_update_n = 0;
}
#undef CHUNKS
#undef VOXEL_WORLD
#undef PLAYER

#define SETTINGS deref(settings_ptr)
#define INPUT deref(input_ptr)
#define PLAYER deref(globals_ptr).player
#define VOXEL_WORLD deref(globals_ptr).voxel_world
#define INDIRECT deref(globals_ptr).indirect_dispatch
void voxel_world_perframe(
    daxa_BufferPtr(GpuSettings) settings_ptr,
    daxa_BufferPtr(GpuInput) input_ptr,
    daxa_RWBufferPtr(GpuGlobals) globals_ptr) {

    VOXEL_WORLD.chunk_update_n = 0;

    INDIRECT.chunk_edit_dispatch = u32vec3(CHUNK_SIZE / 8, CHUNK_SIZE / 8, 0);
    INDIRECT.subchunk_x2x4_dispatch = u32vec3(1, 64, 0);
    INDIRECT.subchunk_x8up_dispatch = u32vec3(1, 1, 0);
}
#undef INDIRECT
#undef VOXEL_WORLD
#undef PLAYER
#undef INPUT
#undef SETTINGS

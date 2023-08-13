#pragma once

#include <shared/app.inl>

#include <utils/math.glsl>
#include <voxels/voxels.glsl>

void voxel_world_startup(
    daxa_RWBufferPtr(GpuGlobals) globals_ptr) {
    deref(globals_ptr).voxel_world.chunk_update_n = 0;
}

void voxel_world_perframe(
    daxa_BufferPtr(GpuInput) input_ptr,
    daxa_RWBufferPtr(GpuGlobals) globals_ptr) {

    for (u32 i = 0; i < MAX_CHUNK_UPDATES_PER_FRAME; ++i) {
        deref(globals_ptr).voxel_world.chunk_update_infos[i].brush_flags = 0;
        deref(globals_ptr).voxel_world.chunk_update_infos[i].i = INVALID_CHUNK_I;
    }

    deref(globals_ptr).voxel_world.chunk_update_n = 0;

    deref(globals_ptr).indirect_dispatch.chunk_edit_dispatch = u32vec3(CHUNK_SIZE / 8, CHUNK_SIZE / 8, 0);
    deref(globals_ptr).indirect_dispatch.subchunk_x2x4_dispatch = u32vec3(1, 64, 0);
    deref(globals_ptr).indirect_dispatch.subchunk_x8up_dispatch = u32vec3(1, 1, 0);
}

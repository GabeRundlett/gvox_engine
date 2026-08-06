#pragma once

#include <voxels/voxels.inl>
#include <voxels/brushes.inl>
#include <utilities/gpu/math.glsl>
#include <voxels/pack_unpack.glsl>

struct VoxelTraceResult {
    float dist;
    vec3 nrm;
    vec3 vel;
    uint step_n;
    PackedVoxel voxel_data;
};

#define INVALID_CHUNK_I ivec3(0x80000000)

// 3D Leaf Chunk index => uint index in buffer
uint calc_chunk_index(daxa_BufferPtr(VoxelWorldGlobals) voxel_globals, uvec3 chunk_i, uvec3 chunk_n) {
    // Modulate the chunk index to be wrapped around relative to the chunk offset provided.
    chunk_i = uvec3((ivec3(chunk_i) + (deref(voxel_globals).offset >> ivec3(6 + LOG2_VOXEL_SIZE))) % ivec3(chunk_n));
    uint chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
    return chunk_index;
}

uint calc_chunk_index_from_worldspace(ivec3 chunk_i, uvec3 chunk_n) {
    chunk_i = chunk_i % ivec3(chunk_n) + ivec3(chunk_i.x < 0, chunk_i.y < 0, chunk_i.z < 0) * ivec3(chunk_n);
    uint chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
    return chunk_index;
}

uint calc_palette_region_index(uvec3 inchunk_voxel_i) {
    uvec3 palette_region_i = inchunk_voxel_i / PALETTE_REGION_SIZE;
    return palette_region_i.x + palette_region_i.y * PALETTES_PER_CHUNK_AXIS + palette_region_i.z * PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS;
}

uint calc_palette_voxel_index(uvec3 inchunk_voxel_i) {
    uvec3 palette_voxel_i = inchunk_voxel_i & (PALETTE_REGION_SIZE - 1);
    return palette_voxel_i.x + palette_voxel_i.y * PALETTE_REGION_SIZE + palette_voxel_i.z * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE;
}

PackedVoxel sample_temp_voxel_chunk(
    daxa_BufferPtr(VoxelWorldGlobals) voxel_globals,
    daxa_BufferPtr(VoxelLeafChunk) voxel_chunks_ptr,
    daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunks,
    uvec3 chunk_n, uvec3 voxel_i) {

    uvec3 chunk_i = voxel_i / CHUNK_SIZE;
    uvec3 inchunk_voxel_i = voxel_i - chunk_i * CHUNK_SIZE;
    uint chunk_index = calc_chunk_index(voxel_globals, chunk_i, chunk_n);
    daxa_BufferPtr(VoxelLeafChunk) voxel_chunk_ptr = advance(voxel_chunks_ptr, chunk_index);
    uint update_index = deref(voxel_chunk_ptr).update_index;
    if (update_index != 0) {
        daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr = advance(temp_voxel_chunks, update_index - 1);
        return deref(temp_voxel_chunk_ptr).voxels[inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE];
    }
}

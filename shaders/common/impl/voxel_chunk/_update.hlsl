#pragma once

#include "common/interface/voxel_chunk.hlsl"
#include "common/impl/worldgen.hlsl"

#include "common/impl/voxel_chunk/_common.hlsl"

void VoxelChunk::gen(int3 block_offset) {
    float3 block_pos = float3(block_offset) / VOXEL_SCL + box.bound_min;
    WorldgenState worldgen_state = get_worldgen_state(block_pos);
    block_pass0(worldgen_state, block_pos);
    SurroundingInfo surroundings = get_surrounding(worldgen_state, block_pos);

    block_pass1(worldgen_state, block_pos, surroundings);

    float3 col = block_color(worldgen_state);
    if (worldgen_state.block_id != BlockID::Water && worldgen_state.block_id != BlockID::Air && worldgen_state.block_id != BlockID::Stone)
        col *= rand(block_pos) * 0.2 + 0.8;

    // col = block_pos + surroundings.nrm - float3(0, 0, 16) - worldgen_state.t_noise;

    float3 nrm = terrain_nrm(block_pos);
    // float3 nrm = -surroundings.nrm;
    nrm = nrm * (surroundings.exposure > 0);
    if (worldgen_state.block_id == BlockID::Water)
        nrm = float3(0, 0, 1);

    uint index = calc_index(block_offset);
    Voxel result;
    result.col_id = float4_to_uint(float4(col, 0)) | ((uint)(worldgen_state.block_id) << 0x18);
    result.nrm = float4_to_uint(float4(nrm, 0));

    data[index] = result;
}

void VoxelChunk::do_edit(int3 block_offset, in out EditInfo edit_info) {
    float3 block_pos = float3(block_offset) / VOXEL_SCL + box.bound_min;
    float3 del = block_pos - edit_info.pos;
    float l = length(block_pos - edit_info.pos);

    uint index = calc_index(block_offset);

    if (l < edit_info.radius) {
        // float3 col = float3(1, 0.05, 0.08);
        Voxel result;
        result.col_id = float4_to_uint(float4(edit_info.col, 0)) | ((uint)(edit_info.block_id) << 0x18);
        result.nrm = float4_to_uint(float4(normalize(del) * (l > edit_info.radius - 3.0 / VOXEL_SCL), 0));
        data[index] = result;
    }
}

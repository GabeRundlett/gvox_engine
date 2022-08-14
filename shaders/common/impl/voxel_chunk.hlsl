#pragma once

#include "common/interface/voxel_chunk.hlsl"
#include "common/impl/raytrace.hlsl"
#include "common/impl/worldgen.hlsl"

int3 VoxelChunk::calc_tile_offset(float3 p) {
    return int3((p - box.bound_min) * VOXEL_SCL);
}
uint calc_index(int3 tile_offset) {
    tile_offset = clamp(tile_offset, int3(0, 0, 0), int3(CHUNK_SIZE - 1, CHUNK_SIZE - 1, CHUNK_SIZE - 1));
    return tile_offset.x + tile_offset.y * CHUNK_SIZE + tile_offset.z * CHUNK_SIZE * CHUNK_SIZE;
}
uint VoxelChunk::sample_tile(float3 p) {
    return sample_tile(calc_tile_offset(p));
}
uint VoxelChunk::sample_tile(int3 tile_offset) {
    return sample_tile(calc_index(tile_offset));
}
uint VoxelChunk::sample_tile(uint index) {
    return data[index];
}
float3 VoxelChunk::sample_color(float3 p) {
    // return sample_tile(p) * 0.01;
    return uint_to_float4(sample_tile(p)).rgb;
}

bool is_block_occluding(BlockID block_id) {
    switch (block_id) {
    case BlockID::Air:
        return false;
    default:
        return true;
    }
}

void VoxelChunk::gen(int3 block_offset) {
    float3 block_pos = float3(block_offset) / VOXEL_SCL + box.bound_min;
    WorldgenState worldgen_state = get_worldgen_state(block_pos);
    block_pass0(worldgen_state, block_pos);
    // SurroundingInfo surroundings = get_surrounding(worldgen_state, block_pos);

    SurroundingInfo surroundings;

    for (int i = 0; i < 15; ++i) {
        WorldgenState temp;
        float3 sample_pos;

        sample_pos = block_pos + float3(0, 0, i + 1) / VOXEL_SCL;
        temp = get_worldgen_state(sample_pos);
        block_pass0(temp, sample_pos);
        surroundings.below_ids[i] = temp.block_id;

        sample_pos = block_pos + float3(0, 0, -i - 1) / VOXEL_SCL;
        temp = get_worldgen_state(sample_pos);
        block_pass0(temp, sample_pos);
        surroundings.above_ids[i] = temp.block_id;
    }

    surroundings.depth_above = 0;
    surroundings.depth_below = 0;
    surroundings.above_water = 0;
    surroundings.under_water = 0;

    if (worldgen_state.block_id == BlockID::Air) {
        for (; surroundings.depth_above < 15; ++surroundings.depth_above) {
            if (surroundings.above_ids[surroundings.depth_above] == BlockID::Water)
                surroundings.under_water++;
            if (is_block_occluding(surroundings.above_ids[surroundings.depth_above]))
                break;
        }
        for (; surroundings.depth_below < 15; ++surroundings.depth_below) {
            if (surroundings.below_ids[surroundings.depth_below] == BlockID::Water)
                surroundings.above_water++;
            if (is_block_occluding(surroundings.below_ids[surroundings.depth_below]))
                break;
        }
    } else {
        for (; surroundings.depth_above < 15; ++surroundings.depth_above) {
            if (surroundings.above_ids[surroundings.depth_above] == BlockID::Water)
                surroundings.under_water++;
            if (!is_block_occluding(surroundings.above_ids[surroundings.depth_above]))
                break;
        }
        for (; surroundings.depth_below < 15; ++surroundings.depth_below) {
            if (surroundings.below_ids[surroundings.depth_below] == BlockID::Water)
                surroundings.above_water++;
            if (!is_block_occluding(surroundings.below_ids[surroundings.depth_below]))
                break;
        }
    }
    WorldgenState slope_t0 = get_worldgen_state(block_pos + float3(1, 0, 0) / VOXEL_SCL * 0.01);
    WorldgenState slope_t1 = get_worldgen_state(block_pos + float3(0, 1, 0) / VOXEL_SCL * 0.01);
    WorldgenState slope_t2 = get_worldgen_state(block_pos + float3(0, 0, 1) / VOXEL_SCL * 0.01);
    surroundings.nrm = normalize(float3(slope_t0.t_noise, slope_t1.t_noise, slope_t2.t_noise) - worldgen_state.t_noise);

    block_pass1(worldgen_state, block_pos, surroundings);

    float3 col = block_color(worldgen_state);
    if (worldgen_state.block_id != BlockID::Water && worldgen_state.block_id != BlockID::Air && worldgen_state.block_id != BlockID::Stone)
        col *= rand(block_pos) * 0.2 + 0.8;

    // col = block_pos + surroundings.nrm - float3(0, 0, 16) - worldgen_state.t_noise;

    uint index = calc_index(block_offset);
    data[index] = float4_to_uint(float4(col, 0)) | ((uint)(worldgen_state.block_id) << 0x18);
}

void VoxelChunk::do_edit(int3 block_offset, in out EditInfo edit_info) {
    float3 block_pos = float3(block_offset) / VOXEL_SCL + box.bound_min;
    float l = length(block_pos - edit_info.pos);

    uint index = calc_index(block_offset);

    if (l < edit_info.radius) {
        // float3 col = float3(1, 0.05, 0.08);
        data[index] = float4_to_uint(float4(edit_info.col, 0)) | ((uint)(edit_info.block_id) << 0x18);
    }
}

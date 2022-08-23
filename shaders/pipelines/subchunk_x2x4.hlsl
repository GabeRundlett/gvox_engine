#include "common/buffers.hlsl"

#include "common/impl/voxel_world/_update.hlsl"

struct Push {
    daxa::BufferId globals_id;
    uint mode;
};
[[vk::push_constant]] const Push p;

groupshared uint local_x2_copy[4][4];

void VoxelWorld::subchunk_x2x4(uint3 group_local_id, uint3 group_id, int3 chunk_i) {
    uint chunk_index = get_chunk_index(chunk_i);
    if (chunks_genstate[chunk_index].edit_stage == EditStage::Finished)
        return;

    uint2 x2_in_group_location = uint2(
        (group_local_id.x >> 5) & 0x3,
        (group_local_id.x >> 7) & 0x3);
    uint3 x2_i = uint3(
        (group_local_id.x >> 5) & 0x3,
        (group_local_id.x >> 7) & 0x3,
        (group_local_id.x & 0x1F));
    x2_i += 4 * uint3(group_id.y & 0x7, (group_id.y >> 3) & 0x7, 0);
    uint3 in_chunk_i = x2_i * 2;
    bool at_least_one_occluding = false;
    BlockID base_id_x1 = (BlockID)voxel_chunks[chunk_index].sample_tile(in_chunk_i);
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z) {
                int3 local_i = in_chunk_i + int3(x, y, z); // in x1 space
                at_least_one_occluding = at_least_one_occluding || ((BlockID)voxel_chunks[chunk_index].sample_tile(local_i) != base_id_x1);
            }
    uint result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x2_i);
    }
    uint or_result = WaveActiveBitOr(result);
    if (WaveIsFirstLane()) {
        uint index = uniformity_lod_index<2>(x2_i);
        uniformity_chunks[chunk_index].lod_x2[index] = or_result;
        local_x2_copy[x2_in_group_location.x][x2_in_group_location.y] = or_result;
    }
    GroupMemoryBarrierWithGroupSync();
    if (group_local_id.x >= 64) {
        return;
    }
    uint3 x4_i = uint3(
        (group_local_id.x >> 4) & 0x1,
        (group_local_id.x >> 5) & 0x1,
        group_local_id.x & 0xF);
    x4_i += 2 * uint3(group_id.y & 0x7, (group_id.y >> 3) & 0x7, 0);
    x2_i = x4_i * 2;
    BlockID base_id_x2 = (BlockID)voxel_chunks[chunk_index].sample_tile(x2_i * 2);
    at_least_one_occluding = false;
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z) {
                int3 local_i = x2_i + int3(x, y, z); // in x2 space
                uint mask = uniformity_lod_mask(local_i);
                uint2 x2_in_group_index = uint2(
                    local_i.x & 0x3,
                    local_i.y & 0x3);
                bool is_occluding = (local_x2_copy[x2_in_group_index.x][x2_in_group_index.y] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || is_occluding || ((BlockID)voxel_chunks[chunk_index].sample_tile(local_i * 2) != base_id_x2);
            }
    result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x4_i);
    }
    for (int i = 0; i < 2; i++) {
        if ((WaveGetLaneIndex() >> 4) == i) {
            result = WaveActiveBitOr(result);
        }
    }
    if ((WaveGetLaneIndex() & 0xF /* = %16 */) == 0) {
        uint index = uniformity_lod_index<4>(x4_i);
        uniformity_chunks[chunk_index].lod_x4[index] = result;
    }
}

// clang-format off
[numthreads(512, 1, 1)] void main(uint3 group_local_id: SV_GroupThreadID, uint3 group_id : SV_GroupID) {
    // clang-format on
    StructuredBuffer<Globals> globals = daxa::get_StructuredBuffer<Globals>(p.globals_id);
    uint3 chunk_i = globals[0].game.voxel_world.chunkgen_i;
    // if (p.mode == 1) {
    //     chunk_i += int3(globals[0].game.pick_pos[0].xyz) / CHUNK_SIZE;
    // }
    globals[0].game.voxel_world.subchunk_x2x4(group_local_id, group_id, chunk_i);
}

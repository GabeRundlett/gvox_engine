#include "common/buffers.hlsl"

#include "common/impl/voxel_world/_update.hlsl"

struct Push {
    uint globals_id;
    uint mode;
};
[[vk::push_constant]] const Push p;

groupshared uint local_x8_copy[64];
groupshared uint local_x16_copy[16];

void VoxelWorld::subchunk_x8up(uint3 group_local_id, int3 chunk_i) {
    uint chunk_index = get_chunk_index(chunk_i);
    if (chunks_genstate[chunk_index].edit_stage == EditStage::Finished)
        return;

    uint3 x8_i = uint3(
        (group_local_id.x >> 3) & 0x7,
        (group_local_id.x >> 6) & 0x7,
        group_local_id.x & 0x7);
    uint3 x4_i = x8_i * 2;

    BlockID base_id_x4 = (BlockID)voxel_chunks[chunk_index].sample_tile(x4_i * 4);

    bool at_least_one_occluding = false;
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z) {
                int3 local_i = x4_i + int3(x, y, z); // x4 space
                uint index = uniformity_lod_index<4>(local_i);
                uint mask = uniformity_lod_mask(local_i);
                bool occluding = (uniformity_chunks[chunk_index].lod_x4[index] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || occluding || ((BlockID)voxel_chunks[chunk_index].sample_tile(local_i * 4) != base_id_x4);
            }

    uint result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x8_i);
    }
    for (int i = 0; i < 4; i++) {
        if ((WaveGetLaneIndex() >> 3) == i) {
            result = WaveActiveBitOr(result);
        }
    }
    if ((WaveGetLaneIndex() & 0x7 /* == % 8*/) == 0) {
        uint index = uniformity_lod_index<8>(x8_i);
        uniformity_chunks[chunk_index].lod_x8[index] = result;
        local_x8_copy[index] = result;
    }

    GroupMemoryBarrierWithGroupSync();

    if (group_local_id.x >= 64) {
        return;
    }

    uint3 x16_i = uint3(
        (group_local_id.x >> 2) & 0x3,
        (group_local_id.x >> 4) & 0x3,
        group_local_id.x & 0x3);
    x8_i = x16_i * 2;
    BlockID base_id_x8 = (BlockID)voxel_chunks[chunk_index].sample_tile(x8_i * 8);

    at_least_one_occluding = false;
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z) {
                int3 local_i = x8_i + int3(x, y, z); // x8 space
                uint mask = uniformity_lod_mask(local_i);
                uint index = uniformity_lod_index<8>(local_i);
                bool is_occluding = (local_x8_copy[index] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || is_occluding || ((BlockID)voxel_chunks[chunk_index].sample_tile(local_i * 8) != base_id_x8);
            }

    result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x16_i);
    }
    for (int i = 0; i < 8; i++) {
        if ((WaveGetLaneIndex() >> 2) == i) {
            result = WaveActiveBitOr(result);
        }
    }
    if ((WaveGetLaneIndex() & 0x3) == 0) {
        uint index = uniformity_lod_index<16>(x16_i);
        uniformity_chunks[chunk_index].lod_x16[index] = result;
        local_x16_copy[index] = result;
    }

    GroupMemoryBarrierWithGroupSync();

    if (group_local_id.x >= 8) {
        return;
    }

    uint3 x32_i = uint3(
        (group_local_id.x >> 1) & 0x1,
        (group_local_id.x >> 2) & 0x1,
        group_local_id.x & 0x1);
    x16_i = x32_i * 2;
    BlockID base_id_x16 = (BlockID)voxel_chunks[chunk_index].sample_tile(x16_i * 16);

    at_least_one_occluding = false;
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z) {
                int3 local_i = x16_i + int3(x, y, z); // x16 space
                uint mask = uniformity_lod_mask(local_i);
                uint index = uniformity_lod_index<16>(local_i);
                bool is_occluding = (local_x16_copy[index] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || is_occluding || ((BlockID)voxel_chunks[chunk_index].sample_tile(local_i * 16) != base_id_x16);
            }

    result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x32_i);
    }
    for (int i = 0; i < 16; i++) {
        if ((WaveGetLaneIndex() >> 1) == i) {
            result = WaveActiveBitOr(result);
        }
    }
    if ((WaveGetLaneIndex() & 0x1) == 0) {
        uint index = uniformity_lod_index<32>(x32_i);
        uniformity_chunks[chunk_index].lod_x32[index] = result;
    }
}

// clang-format off
[numthreads(512, 1, 1)] void main(uint3 group_local_id: SV_GroupThreadID) {
    // clang-format on
    StructuredBuffer<Globals> globals = daxa::getBuffer<Globals>(p.globals_id);
    uint3 chunk_i = globals[0].game.voxel_world.chunkgen_i;
    // if (p.mode == 1) {
    //     chunk_i += int3(globals[0].game.pick_pos[0].xyz) / CHUNK_SIZE;
    // }
    globals[0].game.voxel_world.subchunk_x8up(group_local_id, chunk_i);
}

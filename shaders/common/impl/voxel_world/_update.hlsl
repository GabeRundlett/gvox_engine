#pragma once

#include "common/interface/voxel_world.hlsl"

#include "common/impl/voxel_world/_common.hlsl"
#include "common/impl/voxel_chunk/_update.hlsl"

void VoxelWorld::init() {
    for (int zi = 0; zi < CHUNK_COUNT_Z; ++zi) {
        for (int yi = 0; yi < CHUNK_COUNT_Y; ++yi) {
            for (int xi = 0; xi < CHUNK_COUNT_X; ++xi) {
                int index = get_chunk_index(int3(xi, yi, zi));
                voxel_chunks[index].box.bound_min = float3(xi, yi, zi) * (CHUNK_SIZE / VOXEL_SCL);
                voxel_chunks[index].box.bound_max = voxel_chunks[index].box.bound_min + (CHUNK_SIZE / VOXEL_SCL);
            }
        }
    }
    box.bound_min = voxel_chunks[0].box.bound_min;
    box.bound_max = voxel_chunks[CHUNK_N - 1].box.bound_max;

    float min_dist = 1e38;
    for (uint zi = 0; zi < CHUNK_COUNT_Z / 1; ++zi) {
        for (uint yi = 0; yi < CHUNK_COUNT_Y / 1; ++yi) {
            for (uint xi = 0; xi < CHUNK_COUNT_X / 1; ++xi) {
                uint i = get_chunk_index(int3(xi, yi, zi));
                float3 chunk_p = (float3(xi, yi, zi) * 1 + 2) * CHUNK_SIZE / VOXEL_SCL + box.bound_min;
                float chunk_dist = length(chunk_p - center_pt);
                if (chunk_dist < min_dist) {
                    min_dist = chunk_dist;
                    chunkgen_i = int3(xi, yi, zi);
                }
            }
        }
    }
}

void VoxelWorld::chunkgen(int3 block_offset, StructuredBuffer<GpuInput> input) {
    int3 chunk_i = clamp(chunkgen_i, int3(0, 0, 0), int3(CHUNK_COUNT_X - 1, CHUNK_COUNT_Y - 1, CHUNK_COUNT_Z - 1));
    int index = clamp(get_chunk_index(chunk_i), 0, CHUNK_N - 1);
    switch (chunks_genstate[index].edit_stage) {
    case EditStage::ProceduralGen: voxel_chunks[index].gen(block_offset); break;
    case EditStage::BlockEdit: voxel_chunks[index].do_edit(block_offset, edit_info); break;
    default: break;
    }
}

void VoxelWorld::queue_edit() {
    int3 chunk_i = clamp(int3(edit_info.pos * VOXEL_SCL / CHUNK_SIZE), int3(0, 0, 0), int3(CHUNK_COUNT_X - 1, CHUNK_COUNT_Y - 1, CHUNK_COUNT_Z - 1));
    int index = clamp(get_chunk_index(chunk_i), 0, CHUNK_N - 1);
    chunks_genstate[index].edit_stage = EditStage::BlockEdit;
}

void VoxelWorld::update(StructuredBuffer<GpuInput> input) {
    uint prev_i = get_chunk_index(chunkgen_i);
    if (chunkgen_i.x != -1000)
        chunks_genstate[prev_i].edit_stage = EditStage::Finished;

    bool finished = true;
    float min_dist = 1e38;
    for (uint zi = 0; zi < CHUNK_COUNT_Z / 1; ++zi) {
        for (uint yi = 0; yi < CHUNK_COUNT_Y / 1; ++yi) {
            for (uint xi = 0; xi < CHUNK_COUNT_X / 1; ++xi) {
                uint i = get_chunk_index(int3(xi, yi, zi));
                Box chunk_box = voxel_chunks[i].box;
                if (chunks_genstate[i].edit_stage != EditStage::Finished) {
                    finished = false;
                    float3 chunk_p = (chunk_box.bound_min + chunk_box.bound_max) * 0.5;
                    float chunk_dist = length(chunk_p - center_pt);
                    if (chunk_dist < min_dist) {
                        min_dist = chunk_dist;
                        chunkgen_i = int3(xi, yi, zi);
                    }
                    switch (chunks_genstate[i].edit_stage) {
                    case EditStage::None: chunks_genstate[i].edit_stage = EditStage::ProceduralGen; break;
                    default: break;
                    }
                }
            }
        }
    }
}

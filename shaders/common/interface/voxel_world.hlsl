#pragma once

#include "common/interface/voxel_chunk.hlsl"

#define CHUNK_NX 12
#define CHUNK_NY 12
#define CHUNK_NZ 4
#define CHUNK_N (CHUNK_NX * CHUNK_NY * CHUNK_NZ)
#define BLOCK_NX (CHUNK_NX * CHUNK_SIZE)
#define BLOCK_NY (CHUNK_NY * CHUNK_SIZE)
#define BLOCK_NZ (CHUNK_NZ * CHUNK_SIZE)
#define BLOCK_N (CHUNK_N * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE)

enum EditStage {
    None,
    ProceduralGen,
    BlockEdit,
    Finished,
};

struct GenState {
    EditStage edit_stage;
    void default_init() {
        edit_stage = EditStage::None;
    }
};

struct VoxelWorld {
    Box box;
    int3 chunkgen_i;
    float3 center_pt;

    EditInfo edit_info;

    VoxelChunk voxel_chunks[CHUNK_N];
    VoxelUniformityChunk uniformity_chunks[CHUNK_N];
    GenState chunks_genstate[CHUNK_N];
    // VoxelOctree voxel_octrees[CHUNK_N];

    void default_init() {
        chunkgen_i = -1000;
        center_pt = -1000;
        edit_info.default_init();
        for (int i = 0; i < CHUNK_N; ++i) {
            voxel_chunks[i].default_init();
            uniformity_chunks[i].default_init();
            chunks_genstate[i].default_init();
        }
    }

    void init();
    void update(in out Input input);
    void chunkgen(int3 block_offset, in out Input input);
    void queue_edit();
    void subchunk_x2x4(uint3 group_local_id, uint3 group_id, int3 chunk_i);
    void subchunk_x8up(uint3 group_local_id, int3 chunk_i);

    void trace(in out GameTraceState trace_state, Ray ray);
    void eval_color(in out GameTraceState trace_state);

    uint get_chunk_index(int3 chunk_i);
    uint sample_lod(float3 p, in out int chunk_index);
    TraceRecord dda(Ray ray, in out int chunk_index, in uint max_steps);
};

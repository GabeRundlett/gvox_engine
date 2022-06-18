#pragma once

#include "Daxa.hlsl"

#include "block_info.hlsl"

DAXA_DEFINE_BA_RWTEXTURE3D(uint)
DAXA_DEFINE_BA_TEXTURE3D(uint)

#if USE_NEW_PRESENCE
struct ChunkBlockPresence {
    uint data[32 * 32 * 32 / 8];
};
#else
struct ChunkBlockPresence {
    uint x2[1024];
    uint x4[256];
    uint x8[64];
    uint x16[16];
    uint x32[4];
};
#endif

struct Structure {
    float4 p;
    uint id;
    uint _pad[3];
};

struct ChunkgenData {
    Structure structures[128];
    uint structure_n;
    uint _pad[3];
};

struct Globals {
    float4 pos;
    float4 pick_pos[2];
    int2 frame_dim;
    float time;
    float fov;

    uint texture_index;
    uint empty_chunk_index;
    uint model_load_index;
    uint inventory_index;
    uint chunk_images[CHUNK_NZ][CHUNK_NY][CHUNK_NX];

    // ---- GPU ONLY ----

    ChunkgenData chunkgen_data[CHUNK_NZ][CHUNK_NY][CHUNK_NX];
    ChunkBlockPresence chunk_block_presence[CHUNK_NZ][CHUNK_NY][CHUNK_NX];
};

struct ModelLoadBuffer {
    float4 pos, dim;
    uint data[128 * 128 * 128];
};

DAXA_DEFINE_BA_BUFFER(Globals)
DAXA_DEFINE_BA_BUFFER(ModelLoadBuffer)

BlockID load_block_id(StructuredBuffer<Globals> globals, float3 pos) {
    int3 chunk_i = int3(pos / CHUNK_SIZE);
    if (chunk_i.x < 0 || chunk_i.x > CHUNK_NX - 1 ||
        chunk_i.y < 0 || chunk_i.y > CHUNK_NY - 1 ||
        chunk_i.z < 0 || chunk_i.z > CHUNK_NZ - 1) {
        return BlockID::Air;
    }
    return (BlockID)daxa::getRWTexture3D<uint>(globals[0].chunk_images[chunk_i.z][chunk_i.y][chunk_i.x])
        [int3(pos) - chunk_i * int3(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE)];
}

#include "chunk.hlsl"

struct Push {
    uint4 chunk_i;
    uint globalsID;
    uint mode;
};
[[vk::push_constant]] const Push p;

[numthreads(8, 8, 8)] void main(uint3 global_i
                                : SV_DispatchThreadID) {
    StructuredBuffer<Globals> globals = daxa::getBuffer<Globals>(p.globalsID);
    uint3 chunk_i = p.chunk_i.xyz;
    if (p.mode == 1) {
        chunk_i += int3(globals[0].pick_pos[0].xyz) / CHUNK_SIZE;
    }
    if (chunk_i.x < 0 || chunk_i.x >= CHUNK_NX ||
        chunk_i.y < 0 || chunk_i.y >= CHUNK_NY ||
        chunk_i.z < 0 || chunk_i.z >= CHUNK_NZ)
        return;
    uint chunk_id = globals[0].chunk_images[chunk_i.z][chunk_i.y][chunk_i.x];
    if (chunk_id == globals[0].empty_chunk_index)
        return;

    float3 block_pos = chunk_i * CHUNK_SIZE + global_i * 2;

    uint r = 0;

    {
        bool b0 = is_block_occluding(load_block_id(globals, block_pos + float3(0, 0, 0)));
        bool b1 = is_block_occluding(load_block_id(globals, block_pos + float3(1, 0, 0)));
        bool b2 = is_block_occluding(load_block_id(globals, block_pos + float3(0, 1, 0)));
        bool b3 = is_block_occluding(load_block_id(globals, block_pos + float3(1, 1, 0)));
        bool b4 = is_block_occluding(load_block_id(globals, block_pos + float3(0, 0, 1)));
        bool b5 = is_block_occluding(load_block_id(globals, block_pos + float3(1, 0, 1)));
        bool b6 = is_block_occluding(load_block_id(globals, block_pos + float3(0, 1, 1)));
        bool b7 = is_block_occluding(load_block_id(globals, block_pos + float3(1, 1, 1)));

        uint r0 = b0 || b1 || b2 || b3 || b4 || b5 || b6 || b7;
        r |= (r0 << 0x00);
    }

    uint index = x_index(global_i * 2);
    globals[0].chunk_block_presence[chunk_i.z][chunk_i.y][chunk_i.x].data[index] = r;
}

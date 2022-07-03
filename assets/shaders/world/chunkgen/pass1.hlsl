#include "world/common.hlsl"
#include "world/chunkgen/noise.hlsl"

#include "utils/shape_dist.hlsl"

#include "core.hlsl"

uint gen_block(float3 b_pos) {
    StructuredBuffer<Globals> globals = daxa::getBuffer<Globals>(p.globals_sb);
    int3 chunk_i = p.pos.xyz / CHUNK_SIZE;
    int3 chunk_i_min = int3(max(chunk_i.x - 1, 0), max(chunk_i.y - 1, 0), max(chunk_i.z - 1, 0));
    int3 chunk_i_max = int3(min(chunk_i.x + 1, CHUNK_NX), min(chunk_i.y + 3, CHUNK_NY), min(chunk_i.z + 1, CHUNK_NZ));

    BlockID result;
    result = BlockID::Debug;

    for (uint cz = chunk_i_min.z; cz < chunk_i_max.z; ++cz) {
        for (uint cy = chunk_i_min.y; cy < chunk_i_max.y; ++cy) {
            for (uint cx = chunk_i_min.x; cx < chunk_i_max.x; ++cx) {
                uint structure_n = globals[0].chunkgen_data[cz][cy][cx].structure_n;
                for (uint i = 0; i < min(structure_n, 127); ++i) {
                    float3 p = b_pos - globals[0].chunkgen_data[cz][cy][cx].structures[i].p.xyz;
                    switch (globals[0].chunkgen_data[cz][cy][cx].structures[i].id) {
                    case 1: {
                        p += float3(0, -1, 0);
                        TreeSDF tree_sdf = sd_tree_spruce(p / GEN_SCL, 0);
                        if (tree_sdf.trunk_dist < 0)
                            result = BlockID::Log;
                        if (tree_sdf.leaves_dist < 0)
                            result = BlockID::Leaves;
                    } break;
                    case 2: {
                        p += float3(0, -5, 0);
                        TreeSDF tree_sdf = sd_tree_pine(p / GEN_SCL, 0);
                        if (tree_sdf.trunk_dist < 0)
                            result = BlockID::Log;
                        if (tree_sdf.leaves_dist < 0)
                            result = BlockID::Leaves;
                    } break;
                    case 3: {
                        p += float3(0, 0, 0);
                        TreeSDF cactus_sdf = sd_cactus(p / GEN_SCL, 0);
                        if (cactus_sdf.trunk_dist < 0)
                            result = BlockID::Cactus;
                    } break;
                    // case 4: {
                    //     p += float3(0, 0, 0);
                    //     TreeSDF cactus_sdf = sd_rust_spike(p / GEN_SCL, 0);
                    //     if (cactus_sdf.trunk_dist < 0)
                    //         result = BlockID::Cobblestone;
                    // } break;
                    }
                }
            }
        }
    }
    return (uint)result;
}

[numthreads(8, 8, 8)] void main(uint3 global_i
                                : SV_DispatchThreadID) {
    float3 block_pos = float3(global_i) + p.pos.xyz;

    uint chunk_texture_id = p.output_image_i;
    RWTexture3D<uint> chunk = daxa::getRWTexture3D<uint>(chunk_texture_id);

    uint new_id = gen_block(block_pos * GEN_SCL + BLOCK_OFFSET);
    if (new_id == 0)
        new_id = chunk[int3(global_i)];
    chunk[int3(global_i)] = new_id;
}

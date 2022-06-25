#include "world/common.hlsl"
#include "world/chunkgen/noise.hlsl"

#include "utils/shape_dist.hlsl"

#include "core.hlsl"

void biome_pass0(in out WorldgenState worldgen_state, in float3 b_pos) {
    worldgen_state.biome_id = BiomeID::Plains;
    if (b_pos.y > MIN_WATER_LEVEL + worldgen_state.r * 20) {
        worldgen_state.biome_id = BiomeID::Underworld;
    } else if (worldgen_state.b_noise < 0.7) {
        worldgen_state.biome_id = BiomeID::Forest;
    } else if (worldgen_state.b_noise > 1.5) {
        worldgen_state.biome_id = BiomeID::Desert;
    }
}

void block_pass0(in out WorldgenState worldgen_state, in float3 b_pos) {
    worldgen_state.block_id = BlockID::Air;
    if (b_pos.y > MIN_WATER_LEVEL) {
        if (worldgen_state.u_noise > 0) {
            worldgen_state.block_id = BlockID::Stone;
        } else if (b_pos.y > LAVA_LEVEL) {
            worldgen_state.block_id = BlockID::Lava;
        }
    } else if (worldgen_state.t_noise > 0) {
        worldgen_state.block_id = BlockID::Stone;
    } else if (b_pos.y > WATER_LEVEL) {
        worldgen_state.block_id = BlockID::Water;
    }
}

struct SurroundingInfo {
    BlockID above_ids[15];
    BlockID below_ids[15];
    uint depth_above;
    uint depth_below;
    uint under_water;
    uint above_water;
};

SurroundingInfo get_surrounding(in out WorldgenState worldgen_state,
                                in float3 b_pos) {
    SurroundingInfo result;

    for (int i = 0; i < 15; ++i) {
        WorldgenState temp;
        float3 sample_pos;

        sample_pos = b_pos + float3(0, i + 1, 0) * GEN_SCL;
        temp = get_worldgen_state(sample_pos);
        block_pass0(temp, sample_pos);
        result.below_ids[i] = temp.block_id;

        sample_pos = b_pos + float3(0, -i - 1, 0) * GEN_SCL;
        temp = get_worldgen_state(sample_pos);
        block_pass0(temp, sample_pos);
        result.above_ids[i] = temp.block_id;
    }

    result.depth_above = 0;
    result.depth_below = 0;
    result.above_water = 0;
    result.under_water = 0;

    if (worldgen_state.block_id == BlockID::Air) {
        for (; result.depth_above < 15; ++result.depth_above) {
            if (result.above_ids[result.depth_above] == BlockID::Water)
                result.under_water++;
            if (is_block_occluding(result.above_ids[result.depth_above]))
                break;
        }
        for (; result.depth_below < 15; ++result.depth_below) {
            if (result.below_ids[result.depth_below] == BlockID::Water)
                result.above_water++;
            if (is_block_occluding(result.below_ids[result.depth_below]))
                break;
        }
    } else {
        for (; result.depth_above < 15; ++result.depth_above) {
            if (result.above_ids[result.depth_above] == BlockID::Water)
                result.under_water++;
            if (!is_block_occluding(result.above_ids[result.depth_above]))
                break;
        }
        for (; result.depth_below < 15; ++result.depth_below) {
            if (result.below_ids[result.depth_below] == BlockID::Water)
                result.above_water++;
            if (!is_block_occluding(result.below_ids[result.depth_below]))
                break;
        }
    }

    return result;
}

void block_pass2(in out WorldgenState worldgen_state, in float3 b_pos,
                 in SurroundingInfo surroundings) {
    StructuredBuffer<Globals> globals = daxa::getBuffer<Globals>(p.globals_sb);
    uint3 chunk_i = p.pos.xyz / CHUNK_SIZE;

    if (!is_transparent(worldgen_state.block_id)) {
        switch (worldgen_state.biome_id) {
        case BiomeID::Beach:
            worldgen_state.block_id = BlockID::Sand;
            break;
        case BiomeID::Caves:
            break;
        case BiomeID::Underworld:
            if (surroundings.depth_above < worldgen_state.r * 4) {
                if (worldgen_state.block_id != BlockID::Lava) {
                    if (b_pos.y > LAVA_LEVEL - 2 + worldgen_state.r * 4)
                        worldgen_state.block_id = BlockID::MoltenRock;
                    else
                        worldgen_state.block_id = BlockID::Bedrock;
                }
            } else if (surroundings.depth_below < worldgen_state.r * 4) {
                worldgen_state.block_id = BlockID::Bedrock;
            }
            break;
        case BiomeID::Plains:
        case BiomeID::Forest:
            if (worldgen_state.block_id == BlockID::Stone) {
                if (surroundings.depth_above == 0 && !surroundings.under_water) {
                    worldgen_state.block_id = BlockID::Grass;
                } else if (surroundings.depth_below < worldgen_state.r * 2) {
                    worldgen_state.block_id = BlockID::Cobblestone;
                } else if (surroundings.depth_above + surroundings.under_water < 4 + worldgen_state.r * 6) {
                    worldgen_state.block_id = BlockID::Dirt;
                } else if (surroundings.depth_above + surroundings.under_water < 15 - worldgen_state.r * 10) {
                    if (worldgen_state.r < 0.2)
                        worldgen_state.block_id = BlockID::Dirt;
                    else if (worldgen_state.r < 0.4)
                        worldgen_state.block_id = BlockID::Gravel;
                    else if (worldgen_state.r < 0.6)
                        worldgen_state.block_id = BlockID::Cobblestone;
                }
            }
            break;
        case BiomeID::Desert:
            if (surroundings.depth_above < 2 + worldgen_state.r * 2) {
                worldgen_state.block_id = BlockID::Sand;
            } else if (surroundings.depth_above < 4 + worldgen_state.r * 6) {
                worldgen_state.block_id = BlockID::Sandstone;
            } else if (surroundings.depth_above < 15) {
                if (worldgen_state.r < 0.1)
                    worldgen_state.block_id = BlockID::Sand;
                else if (worldgen_state.r < 0.2)
                    worldgen_state.block_id = BlockID::Gravel;
            }
            break;
        }
    } else if (worldgen_state.block_id == BlockID::Air &&
               !surroundings.above_water) {
        switch (worldgen_state.biome_id) {
        case BiomeID::Plains:
            if (surroundings.depth_below == 0) {
                if (worldgen_state.r < 0.10) {
                    worldgen_state.block_id = BlockID::TallGrass;
                } else if (worldgen_state.r < 0.11) {
                    worldgen_state.block_id = BlockID::Leaves;
                }
            }
            break;
        case BiomeID::Forest:
            if (worldgen_state.r_xz < 0.01) {
                int trunk_height = 1; // int(5 + worldgen_state.r_xz * 400);
#if INJECT_STRUCTURES
                if (surroundings.depth_below < trunk_height) {
                    if (globals[0].chunkgen_data[chunk_i.z][chunk_i.y][chunk_i.x].structure_n < 127) {
                        int structure_n;
                        InterlockedAdd(globals[0].chunkgen_data[chunk_i.z][chunk_i.y][chunk_i.x].structure_n, 1, structure_n);
                        globals[0].chunkgen_data[chunk_i.z][chunk_i.y][chunk_i.x].structures[structure_n].p = float4(b_pos, 0);
                        globals[0].chunkgen_data[chunk_i.z][chunk_i.y][chunk_i.x].structures[structure_n].id = (int(b_pos.x + 10000) % 5 == 0) ? 2 : 1;
                    }
                    worldgen_state.block_id = BlockID::Log;
                }
#endif
            } else if (worldgen_state.r < 0.5 && surroundings.depth_below == 0) {
                worldgen_state.block_id = BlockID::Leaves;
            }
            break;
        case BiomeID::Desert:
            if (worldgen_state.r_xz < 0.005) {
                int trunk_height = int(5 + worldgen_state.r_xz * 400);
                if (surroundings.depth_below < trunk_height) {
                    worldgen_state.block_id = BlockID::Cactus;
                }
            } else if (worldgen_state.r < 0.02 && surroundings.depth_below == 0) {
                worldgen_state.block_id = BlockID::DriedShrub;
            }
            break;
        default:
            break;
        }
    }
}

BlockID gen_block(in float3 b_pos) {
    WorldgenState worldgen_state = get_worldgen_state(b_pos);

    biome_pass0(worldgen_state, b_pos);

#if VISUALIZE_BIOMES
    switch (worldgen_state.biome_id) {
    case BiomeID::Beach: worldgen_state.block_id = BlockID::Water; break;
    case BiomeID::Caves: worldgen_state.block_id = BlockID::Cobblestone; break;
    case BiomeID::Underworld: worldgen_state.block_id = BlockID::Bedrock; break;
    case BiomeID::Plains: worldgen_state.block_id = BlockID::TallGrass; break;
    case BiomeID::Forest: worldgen_state.block_id = BlockID::Log; break;
    case BiomeID::Desert: worldgen_state.block_id = BlockID::Sand; break;
    }
#else
    block_pass0(worldgen_state, b_pos);
#if DO_DETAILED_GEN
    SurroundingInfo surroundings = get_surrounding(worldgen_state, b_pos);
    block_pass2(worldgen_state, b_pos, surroundings);
#endif
#endif

    return worldgen_state.block_id;
}

[numthreads(8, 8, 8)] void main(uint3 global_i
                                : SV_DispatchThreadID) {
    float3 block_pos = float3(global_i) + p.pos.xyz;

    uint chunk_texture_id = p.output_image_i;
    RWTexture3D<uint> chunk = daxa::getRWTexture3D<uint>(chunk_texture_id);

    chunk[int3(global_i)] = (uint)gen_block(block_pos * GEN_SCL * float3(1, 1, 1) + BLOCK_OFFSET);
}

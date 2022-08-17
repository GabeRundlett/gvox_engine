#pragma once

#include "common/interface/worldgen.hlsl"

#include "utils/noise.hlsl"

float3 block_color(in WorldgenState worldgen_state) {
    // clang-format off
    switch (worldgen_state.block_id) {
    case BlockID::Debug:           return float3(0.60, 0.00, 0.50); break;
    case BlockID::Air:             return float3(0.66, 0.67, 0.91); break;
    case BlockID::Bedrock:         return float3(0.30, 0.30, 0.30); break;
    case BlockID::Brick:           return float3(0.47, 0.23, 0.20); break;
    case BlockID::Cactus:          return float3(0.36, 0.62, 0.28); break;
    case BlockID::Cobblestone:     return float3(0.32, 0.31, 0.31); break;
    case BlockID::CompressedStone: return float3(0.32, 0.31, 0.31); break;
    case BlockID::DiamondOre:      return float3(0.18, 0.67, 0.69); break;
    case BlockID::Dirt:            return float3(0.08, 0.05, 0.03); break;
    case BlockID::DriedShrub:      return float3(0.52, 0.36, 0.27); break;
    case BlockID::Grass:           return float3(0.05, 0.09, 0.03); break;
    case BlockID::Gravel:          return float3(0.10, 0.08, 0.07); break;
    case BlockID::Lava:            return float3(0.00, 0.00, 0.00); break;
    case BlockID::Leaves:          return float3(0.10, 0.29, 0.10); break;
    case BlockID::Log:             return float3(0.23, 0.14, 0.10); break;
    case BlockID::MoltenRock:      return float3(0.20, 0.13, 0.12); break;
    case BlockID::Planks:          return float3(0.68, 0.47, 0.35); break;
    case BlockID::Rose:            return float3(0.52, 0.04, 0.05); break;
    case BlockID::Sand:            return float3(0.82, 0.50, 0.19); break;
    case BlockID::Sandstone:       return float3(0.94, 0.65, 0.38); break;
    case BlockID::Stone:           return float3(0.10, 0.09, 0.08); break;
    case BlockID::TallGrass:       return float3(0.04, 0.08, 0.03); break;
    case BlockID::Water:           return float3(0.10, 0.18, 0.93); break;
    default:                       return float3(0.00, 0.00, 0.00); break;
    }
    // clang-format on
}

bool is_transparent(BlockID block_id) {
    switch (block_id) {
    case BlockID::Air:
    case BlockID::Water:
        return true;
    default:
        return false;
    }
}

bool is_block_occluding(BlockID block_id) {
    switch (block_id) {
    case BlockID::Air:
        return false;
    default:
        return true;
    }
}

float terrain_noise(float3 pos) {
    FractalNoiseConfig noise_conf = {
        /* .amplitude   = */ 1.0,
        /* .persistance = */ 0.12,
        /* .scale       = */ 0.05,
        /* .lacunarity  = */ 4.7,
        /* .octaves     = */ 4,
    };
    float val = fractal_noise(pos, noise_conf);
    val = val - (pos.z - 20) * 0.1;
    val += smoothstep(-1, 1, -(pos.z - 20 + 4) * 2.5) * 0.1;
    val = max(val, 0);
    return val;
}

WorldgenState get_worldgen_state(float3 block_pos) {
    WorldgenState worldgen_state;
    worldgen_state.t_noise = terrain_noise(block_pos);
    worldgen_state.block_id = BlockID::Air;
    worldgen_state.r = rand(block_pos);
    worldgen_state.r_xy = rand(block_pos * float3(1, 1, 0) + 1000);
    return worldgen_state;
}

void block_pass0(in out WorldgenState worldgen_state, float3 block_pos) {
    if (worldgen_state.t_noise > 0) {
        worldgen_state.block_id = BlockID::Stone;
    }
}

float3 terrain_nrm(in float3 pos) {
    float2 e = float2(1.0, -1.0) * 0.5773;
    const float eps = 0.001;
    return -normalize(
        e.xyy * terrain_noise(pos + e.xyy * eps) +
        e.yyx * terrain_noise(pos + e.yyx * eps) +
        e.yxy * terrain_noise(pos + e.yxy * eps) +
        e.xxx * terrain_noise(pos + e.xxx * eps));
}

void block_pass1(in out WorldgenState worldgen_state, float3 block_pos, in SurroundingInfo surroundings) {
    float rough_depth = worldgen_state.t_noise;
    float upwards = max(dot(surroundings.nrm, float3(0, 0, -1)), 0);

    if (worldgen_state.block_id == BlockID::Air) {
        if (block_pos.z < 16) {
            worldgen_state.block_id = BlockID::Water;
        }
    } else {
        float3 p = block_pos - float3(0, 0, 16) + surroundings.nrm * max(dot(surroundings.nrm, float3(0, 0, -1)) - worldgen_state.t_noise * 10, 0);
        if (p.z < 0) {
            if (upwards - rough_depth * 15 > 0) {
                worldgen_state.block_id = BlockID::Sand;
            } else if (upwards - rough_depth * 9 - worldgen_state.r * 0.1 > 0) {
                worldgen_state.block_id = BlockID::Sandstone;
            } else if (upwards - rough_depth * 5 - worldgen_state.r * 0.1 > 0) {
                worldgen_state.block_id = BlockID::Gravel;
            }
        } else {
            if (upwards - rough_depth * 8 > 0) {
                if (surroundings.depth_below < 1) {
                    switch (int(worldgen_state.r * 2)) {
                    case 0: worldgen_state.block_id = BlockID::Grass; break;
                    case 1: worldgen_state.block_id = BlockID::TallGrass; break;
                    }
                } else {
                    worldgen_state.block_id = BlockID::Dirt;
                }
            } else if (upwards - rough_depth * 5 - worldgen_state.r * 0.1 > 0) {
                worldgen_state.block_id = BlockID::Gravel;
            }
        }
    }
}

SurroundingInfo get_surrounding(in out WorldgenState worldgen_state, float3 block_pos) {
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

    float3 neighbor_offsets[] = {
        float3(+1, +0, +0) / VOXEL_SCL,
        float3(-1, +0, +0) / VOXEL_SCL,
        float3(+0, +1, +0) / VOXEL_SCL,
        float3(+0, -1, +0) / VOXEL_SCL,
        float3(+0, +0, +1) / VOXEL_SCL,
        float3(+0, +0, -1) / VOXEL_SCL,
    };

    surroundings.exposure = 0;
    for (int i = 0; i < 6; ++i)
    {
        float3 sample_pos = block_pos + neighbor_offsets[i];
        WorldgenState temp = get_worldgen_state(sample_pos);
        block_pass0(temp, sample_pos);
        surroundings.neighbor_ids[i] = temp.block_id;
        if (!is_block_occluding(temp.block_id))
            ++surroundings.exposure;
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

    return surroundings;
}

#pragma once

#include <utils/noise.glsl>
#include <utils/voxel.glsl>

#define WATER_LEVEL 4

f32 terrain_noise(f32vec3 p) {
    FractalNoiseConfig noise_conf = FractalNoiseConfig(
        /* .amplitude   = */ INPUT.settings.gen_amplitude,
        /* .persistance = */ INPUT.settings.gen_persistance,
        /* .scale       = */ INPUT.settings.gen_scale,
        /* .lacunarity  = */ INPUT.settings.gen_lacunarity,
        /* .octaves     = */ INPUT.settings.gen_octaves);
    f32 val = fractal_noise(p, noise_conf);
    val = val - (p.z - WATER_LEVEL - 4) * 0.02 / 8 * VOXEL_SCL;
    val += smoothstep(-1, 1, -atan((p.z - WATER_LEVEL + 0) * 0.5 / 8 * VOXEL_SCL)) * 0.5;
    val = max(val, 0);
    return val;
}

WorldgenState get_worldgen_state(f32vec3 voxel_p) {
    WorldgenState worldgen_state;
    worldgen_state.t_noise = terrain_noise(voxel_p);
    worldgen_state.block_id = 0;
    worldgen_state.r = rand(voxel_p);
    worldgen_state.r_xy = rand(voxel_p * f32vec3(1, 1, 0) + 1000);
    return worldgen_state;
}

void block_pass0(in out WorldgenState worldgen_state, f32vec3 voxel_p) {
    if (worldgen_state.t_noise > 0 || voxel_p.z < 0.3) {
        worldgen_state.block_id = BlockID_Stone;
    } else {
        worldgen_state.block_id = BlockID_Air;
    }
}

void block_pass1(in out WorldgenState worldgen_state, f32vec3 voxel_p, in SurroundingInfo surroundings) {
    f32 rough_depth = worldgen_state.t_noise;
    f32 upwards = max(dot(surroundings.nrm, f32vec3(0, 0, -1)), 0);

    if (worldgen_state.block_id == BlockID_Air) {
        if (voxel_p.z < WATER_LEVEL) {
            worldgen_state.block_id = BlockID_Water;
        }
    } else {
        f32vec3 p = voxel_p - f32vec3(0, 0, WATER_LEVEL) + surroundings.nrm * max(dot(surroundings.nrm, f32vec3(0, 0, -1)) - worldgen_state.t_noise * 10, 0);
        if (p.z < 0) {
            if (upwards - rough_depth * 15 > 0) {
                worldgen_state.block_id = BlockID_Sand;
            } else if (upwards - rough_depth * 9 - worldgen_state.r * 0.1 > 0) {
                worldgen_state.block_id = BlockID_Sandstone;
            } else if (upwards - rough_depth * 5 - worldgen_state.r * 0.1 > 0) {
                worldgen_state.block_id = BlockID_Gravel;
            }
        } else {
            if (upwards - rough_depth * 8 > 0) {
                if (surroundings.depth_below < 1) {
                    switch (int(worldgen_state.r * 2)) {
                    case 0: worldgen_state.block_id = BlockID_Grass; break;
                    case 1: worldgen_state.block_id = BlockID_TallGrass; break;
                    }
                } else {
                    worldgen_state.block_id = BlockID_Dirt;
                }
            } else if (upwards - rough_depth * 5 - worldgen_state.r * 0.1 > 0) {
                worldgen_state.block_id = BlockID_Gravel;
            }
        }
    }
}

b32 is_block_occluding(u32 block_id) {
    switch (block_id) {
    case BlockID_Air:
        return false;
    default:
        return true;
    }
}

Voxel gen_voxel(in f32vec3 voxel_p) {
    Voxel result;

    voxel_p = voxel_p - INPUT.settings.gen_origin;

    WorldgenState worldgen_state = get_worldgen_state(voxel_p);
    block_pass0(worldgen_state, voxel_p);

    SurroundingInfo surroundings;

    for (int i = 0; i < 15; ++i) {
        WorldgenState temp;
        f32vec3 sample_pos;

        sample_pos = voxel_p + f32vec3(0, 0, i + 1) / VOXEL_SCL;
        temp = get_worldgen_state(sample_pos);
        block_pass0(temp, sample_pos);
        surroundings.below_ids[i] = temp.block_id;

        sample_pos = voxel_p + f32vec3(0, 0, -i - 1) / VOXEL_SCL;
        temp = get_worldgen_state(sample_pos);
        block_pass0(temp, sample_pos);
        surroundings.above_ids[i] = temp.block_id;
    }

    surroundings.depth_above = 0;
    surroundings.depth_below = 0;
    surroundings.above_water = 0;
    surroundings.under_water = 0;

    if (worldgen_state.block_id == BlockID_Air) {
        for (; surroundings.depth_above < 15; ++surroundings.depth_above) {
            if (surroundings.above_ids[surroundings.depth_above] == BlockID_Water)
                surroundings.under_water++;
            if (is_block_occluding(surroundings.above_ids[surroundings.depth_above]))
                break;
        }
        for (; surroundings.depth_below < 15; ++surroundings.depth_below) {
            if (surroundings.below_ids[surroundings.depth_below] == BlockID_Water)
                surroundings.above_water++;
            if (is_block_occluding(surroundings.below_ids[surroundings.depth_below]))
                break;
        }
    } else {
        for (; surroundings.depth_above < 15; ++surroundings.depth_above) {
            if (surroundings.above_ids[surroundings.depth_above] == BlockID_Water)
                surroundings.under_water++;
            if (!is_block_occluding(surroundings.above_ids[surroundings.depth_above]))
                break;
        }
        for (; surroundings.depth_below < 15; ++surroundings.depth_below) {
            if (surroundings.below_ids[surroundings.depth_below] == BlockID_Water)
                surroundings.above_water++;
            if (!is_block_occluding(surroundings.below_ids[surroundings.depth_below]))
                break;
        }
    }
    WorldgenState slope_t0 = get_worldgen_state(voxel_p + f32vec3(1, 0, 0) / VOXEL_SCL * 0.01);
    WorldgenState slope_t1 = get_worldgen_state(voxel_p + f32vec3(0, 1, 0) / VOXEL_SCL * 0.01);
    WorldgenState slope_t2 = get_worldgen_state(voxel_p + f32vec3(0, 0, 1) / VOXEL_SCL * 0.01);
    surroundings.nrm = normalize(f32vec3(slope_t0.t_noise, slope_t1.t_noise, slope_t2.t_noise) - worldgen_state.t_noise);

    block_pass1(worldgen_state, voxel_p, surroundings);

    result.col = f32vec3(1, 0.5, 0.8);
    result.nrm = surroundings.nrm;
    result.block_id = worldgen_state.block_id;

    return result;
}

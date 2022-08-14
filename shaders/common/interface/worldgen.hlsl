#pragma once

enum class BlockID : uint {
    Debug,
    Air,
    Bedrock,
    Brick,
    Cactus,
    Cobblestone,
    CompressedStone,
    DiamondOre,
    Dirt,
    DriedShrub,
    Grass,
    Gravel,
    Lava,
    Leaves,
    Log,
    MoltenRock,
    Planks,
    Rose,
    Sand,
    Sandstone,
    Stone,
    TallGrass,
    Water,
};

struct WorldgenState {
    float t_noise;
    float r, r_xy;
    BlockID block_id;
};

struct SurroundingInfo {
    BlockID above_ids[15];
    BlockID below_ids[15];
    uint depth_above;
    uint depth_below;
    uint under_water;
    uint above_water;
    float3 nrm;
};

float3 block_color(in WorldgenState worldgen_state);
bool is_transparent(BlockID block_id);
float terrain_noise(float3 pos);
WorldgenState get_worldgen_state(float3 pos);
void block_pass0(in out WorldgenState worldgen_state, float3 block_pos);
void block_pass1(in out WorldgenState worldgen_state, float3 block_pos, in SurroundingInfo surroundings);

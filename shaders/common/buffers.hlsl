#pragma once

#include "Daxa.hlsl"

#include "common/interface/game.hlsl"

DAXA_DEFINE_BA_BUFFER(Input);

struct Globals {
    Game game;
};
DAXA_DEFINE_BA_BUFFER(Globals);

struct Readback {
    float3 player_pos;
    uint _pad0;
};
DAXA_DEFINE_BA_BUFFER(Readback);

#pragma once

#include "daxa/daxa.hlsl"

#include "common/interface/game.hlsl"

DAXA_DEFINE_GET_STRUCTURED_BUFFER(Input);

struct Globals {
    Game game;
};
DAXA_DEFINE_GET_STRUCTURED_BUFFER(Globals);

#pragma once

#include "common/interface/game.hlsl"

struct GpuGlobals {
    Game game;
};
DAXA_DEFINE_GET_STRUCTURED_BUFFER(GpuGlobals);

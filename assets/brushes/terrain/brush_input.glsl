#pragma once

DAXA_DECL_BUFFER_STRUCT(CustomBrushSettings, {
    f32vec3 origin;
    f32 amplitude;
    f32 persistance;
    f32 scale;
    f32 lacunarity;
    i32 octaves;
});

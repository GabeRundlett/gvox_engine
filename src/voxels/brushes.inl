#pragma once

struct BrushInput {
    daxa_f32vec3 pos;
    daxa_i32vec3 pos_offset;
    daxa_f32vec3 prev_pos;
    daxa_i32vec3 prev_pos_offset;
};

struct PackedVoxel {
    daxa_u32 data;
};

struct Voxel {
    daxa_u32 material_type; // 2 bits (empty, dielectric, metallic, emissive)
    daxa_f32 roughness;     // 4 bits
    daxa_f32vec3 normal;    // 8 bits
    daxa_f32vec3 color;     // 18 bits
};

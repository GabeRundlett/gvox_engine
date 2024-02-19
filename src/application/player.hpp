#pragma once

#include "input.inl"

struct PlayerInput {
    daxa_u32vec2 frame_dim;
    daxa_f32vec2 halton_jitter;
    daxa_f32 delta_time;
    daxa_f32 sensitivity;
    daxa_f32 fov;
    MouseInput mouse;
    uint32_t actions[GAME_ACTION_LAST + 1];
};

using mat4 = daxa_f32mat4x4;
using vec3 = daxa_f32vec3;

void apply_friction(PlayerInput &INPUT, vec3 &vel, vec3 &friction_vec, float friction_coeff);
void player_fix_chunk_offset(Player &PLAYER);
void player_startup(Player &PLAYER);
void player_perframe(PlayerInput &INPUT, Player &PLAYER);

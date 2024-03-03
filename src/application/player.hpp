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

void apply_friction(PlayerInput &INPUT, daxa_f32vec3 &vel, daxa_f32vec3 &friction_vec, float friction_coeff);
void player_fix_chunk_offset(Player &PLAYER);
void player_startup(Player &PLAYER);
void player_perframe(PlayerInput &INPUT, Player &PLAYER);

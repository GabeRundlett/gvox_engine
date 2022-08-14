#pragma once

#include "common/interface/player.hlsl"
#include "common/interface/voxel_world.hlsl"
#include "common/impl/camera.hlsl"
#include "utils/math.hlsl"

// velocity/s (m/s/s)
#define EARTH_GRAV 9.807
#define MOON_GRAV 1.625
#define MARS_GRAV 3.728
#define JUPITER_GRAV 25.93

#define GRAVITY EARTH_GRAV

// jump height (meters)
#define EARTH_JUMP_HEIGHT 0.59

void Player::init() {
    mouse_sens = 1.0;
    speed = PLAYER_SPEED;
    accel_rate = PLAYER_ACCEL;

    sprint_speed = 2.5f;

    if ((init_flags & 0x01) != 0x01) {
        init_flags = 0x01;
        pos = float3((BLOCK_NX + 0.5) / (2 * VOXEL_SCL), 10, BLOCK_NZ / VOXEL_SCL * 1.1);
        rot = float3(1, 0, 0);
        force_vel = float3(0, 0, 0);
        move_vel = float3(0, 0, 0);
        view_state = 0;
        set_flag_bit(move_flags, 7, true);
    }
}

void Player::toggle_view() {
    view_state = (view_state & ~0xf) | ((view_state & 0xf) + 1);
    if ((view_state & 0xf) > 1)
        view_state = (view_state & ~0xf) | 0;
}

float3 Player::view_vec() {
    switch (view_state) {
    case 1: return float3(0, -2, 0);
    default: return float3(0, 0, 0);
    }
}

void Player::calc_update(in out Input input) {
    update_keys(input);

    if (should_noclip()) {
        speed = PLAYER_SPEED * 10;
        accel_rate = PLAYER_ACCEL * 10;
        on_ground = true;
    } else {
        speed = PLAYER_SPEED;
        accel_rate = PLAYER_ACCEL;
        if (should_move_u() && on_ground)
            force_vel.z = EARTH_GRAV * sqrt(EARTH_JUMP_HEIGHT * 2.0 / EARTH_GRAV);
    }

    if (input.keyboard.keys[GLFW_KEY_F5] != 0) {
        if ((view_state & 0x10) == 0) {
            view_state |= 0x10;
            toggle_view();
        }
    } else {
        view_state &= ~0x10;
    }

    rot.z -= input.mouse.pos_delta.x * mouse_sens * 0.001f;
    rot.x -= input.mouse.pos_delta.y * mouse_sens * 0.001f;
    const float MAX_ROT = deg2rad(90);
    if (rot.x > MAX_ROT)
        rot.x = MAX_ROT;
    if (rot.x < -MAX_ROT)
        rot.x = -MAX_ROT;
    float sin_rot_x = sin(rot.x), cos_rot_x = cos(rot.x);
    float sin_rot_y = sin(rot.y), cos_rot_y = cos(rot.y);
    float sin_rot_z = sin(rot.z), cos_rot_z = cos(rot.z);

    // clang-format off
    camera.rot_mat = mul(
        float3x3(
            cos_rot_z, -sin_rot_z, 0,
            sin_rot_z,  cos_rot_z, 0,
            0,          0,         1
        ),
        float3x3(
            1,          0,          0,
            0,  cos_rot_x,  sin_rot_x,
            0, -sin_rot_x,  cos_rot_x
        ));
    // clang-format on

    float3 move_vec = float3(0, 0, 0);
    float3 forward = float3(-sin_rot_z, +cos_rot_z, 0);
    float3 lateral = float3(+cos_rot_z, +sin_rot_z, 0);

    if (should_move_f())
        move_vec += forward;
    if (should_move_b())
        move_vec -= forward;
    if (should_move_l())
        move_vec -= lateral;
    if (should_move_r())
        move_vec += lateral;

    if (should_move_u())
        move_vec += float3(0, 0, 1);
    if (should_move_d())
        move_vec -= float3(0, 0, 1);

    float applied_accel = accel_rate;
    if (should_sprint())
        max_speed += input.delta_time * accel_rate;
    else
        max_speed -= input.delta_time * accel_rate;

    max_speed = clamp(max_speed, speed, speed * sprint_speed);

    float move_magsq = dot(move_vec, move_vec);
    if (move_magsq > 0) {
        move_vec = normalize(move_vec) * input.delta_time * applied_accel * (on_ground ? 1 : 0.1);
        bobbing_phase += input.delta_time * 18;
    }

    float move_vel_mag = length(move_vel);
    float3 move_vel_dir;
    if (move_vel_mag > 0) {
        move_vel_dir = move_vel / move_vel_mag;
        if (on_ground)
            move_vel -= move_vel_dir * min(accel_rate * 0.4 * input.delta_time, move_vel_mag);
    }

    move_vel += move_vec * 2;

    move_vel_mag = length(move_vel);
    if (move_vel_mag > 0) {
        move_vel = move_vel / move_vel_mag * min(move_vel_mag, max_speed);
    } else {
        bobbing_phase = 0;
    }

    if (should_noclip()) {
        force_vel = 0;
    } else {
        force_vel += float3(0, 0, -1) * GRAVITY * input.delta_time;
    }
}

void Player::apply_update(float3 v, in out Input input) {
    pos += (move_vel + force_vel) * input.delta_time;

    if (pos.z <= 0) {
        pos.z = 0;
        on_ground = true;
        force_vel.z = 0;
        move_vel.z = 0;
    } else {
        on_ground = true;
    }

    prev_camera = camera;

    float move_vel_mag = length(move_vel);
    float bobbing_offset = sin(bobbing_phase) * 0.005 * min(move_vel_mag, 5) * on_ground * (view_state == 0);
    float3 cam_offset = mul(camera.rot_mat, view_vec());
    float3 new_cam_pos = pos + float3(0, 0, PLAYER_EYE_HEIGHT + bobbing_offset) + cam_offset;
    // camera.pos = new_cam_pos * 0.4 + camera.pos * 0.6;
    camera.pos = new_cam_pos;

    camera.update(input);
}

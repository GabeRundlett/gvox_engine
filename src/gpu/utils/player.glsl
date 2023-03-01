#pragma once

#include <shared/shared.inl>

#include <utils/math.glsl>

#define PLAYER deref(globals_ptr).player
void player_startup(daxa_RWBufferPtr(GpuGlobals) globals_ptr) {
    PLAYER.pos = f32vec3(0, 0, 1);
    PLAYER.rot.x = -PI / 2;
}
#undef PLAYER

#define PLAYER deref(globals_ptr).player
#define INPUT deref(input_ptr)
#define SETTINGS deref(settings_ptr)
void player_perframe(daxa_BufferPtr(GpuSettings) settings_ptr, daxa_BufferPtr(GpuInput) input_ptr, daxa_RWBufferPtr(GpuGlobals) globals_ptr) {
    const f32 mouse_sens = 1.0;

    PLAYER.rot.z += INPUT.mouse.pos_delta.x * mouse_sens * SETTINGS.sensitivity * 0.001;
    PLAYER.rot.x -= INPUT.mouse.pos_delta.y * mouse_sens * SETTINGS.sensitivity * 0.001;

    const float MAX_ROT_EPS = 0.01;
    PLAYER.rot.x = clamp(PLAYER.rot.x, MAX_ROT_EPS - PI, -MAX_ROT_EPS);
    float sin_rot_x = sin(PLAYER.rot.x), cos_rot_x = cos(PLAYER.rot.x);
    // float sin_rot_y = sin(PLAYER.rot.y), cos_rot_y = cos(PLAYER.rot.y);
    float sin_rot_z = sin(PLAYER.rot.z), cos_rot_z = cos(PLAYER.rot.z);

    // clang-format off
    PLAYER.cam.prev_rot_mat = PLAYER.cam.rot_mat;
    PLAYER.cam.prev_pos = PLAYER.cam.pos;
    PLAYER.cam.rot_mat = 
        f32mat3x3(
            cos_rot_z, -sin_rot_z, 0,
            sin_rot_z,  cos_rot_z, 0,
            0,          0,         1
        ) *
        // f32mat3x3(
        //     cos_rot_y,  0, sin_rot_y,
        //     0,          1, 0,
        //     -sin_rot_y, 0, cos_rot_y
        // ) *
        f32mat3x3(
            1,          0,          0,
            0,  cos_rot_x,  sin_rot_x,
            0, -sin_rot_x,  cos_rot_x
        );
    // clang-format on
    PLAYER.cam.tan_half_fov = tan(SETTINGS.fov * 0.5);

    f32vec3 move_vec = f32vec3(0, 0, 0);
    PLAYER.forward = f32vec3(+sin_rot_z, +cos_rot_z, 0);
    PLAYER.lateral = f32vec3(+cos_rot_z, -sin_rot_z, 0);

    const f32 accel_rate = 30.0;
    const f32 speed = 10.5;
    const f32 sprint_speed = 2.5;

    if (INPUT.actions[GAME_ACTION_MOVE_FORWARD] != 0)
        move_vec += PLAYER.forward;
    if (INPUT.actions[GAME_ACTION_MOVE_BACKWARD] != 0)
        move_vec -= PLAYER.forward;
    if (INPUT.actions[GAME_ACTION_MOVE_LEFT] != 0)
        move_vec -= PLAYER.lateral;
    if (INPUT.actions[GAME_ACTION_MOVE_RIGHT] != 0)
        move_vec += PLAYER.lateral;

    if (INPUT.actions[GAME_ACTION_JUMP] != 0)
        move_vec += f32vec3(0, 0, 1);
    if (INPUT.actions[GAME_ACTION_CROUCH] != 0)
        move_vec -= f32vec3(0, 0, 1);

    f32 applied_speed = speed;
    if (INPUT.actions[GAME_ACTION_SPRINT] != 0)
        applied_speed *= sprint_speed;

    PLAYER.vel = move_vec * applied_speed;
    PLAYER.pos += PLAYER.vel * INPUT.delta_time;

    PLAYER.cam.pos = PLAYER.pos + f32vec3(0, 0, 0);
}
#undef PLAYER
#undef INPUT
#undef SETTINGS

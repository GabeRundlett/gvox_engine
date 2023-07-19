#pragma once

#include <shared/shared.inl>

#include <utils/math.glsl>

#define PLAYER deref(globals_ptr).player
void player_startup(
    daxa_RWBufferPtr(GpuGlobals) globals_ptr) {
    PLAYER.pos = f32vec3(10.01, 10.02, 80.03);
    PLAYER.chunk_offset = i32vec3(14,14,5);  // TODO: Why is it necessary?
    // PLAYER.pos = f32vec3(150.01, 150.02, 80.03);
    // PLAYER.pos = f32vec3(66.01, 38.02, 14.01);

    // Inside beach hut
    // PLAYER.pos = f32vec3(173.78, 113.72, 12.09);

    PLAYER.rot.x = PI * -0.651;
    PLAYER.rot.z = PI * 0.25;

    // PLAYER.rot.x = PI * -0.751;
    // PLAYER.rot.z = PI * 1.25;
}
#undef PLAYER

#define SETTINGS deref(settings_ptr)
#define INPUT deref(input_ptr)
#define PLAYER deref(globals_ptr).player
void player_perframe(
    daxa_BufferPtr(GpuSettings) settings_ptr,
    daxa_BufferPtr(GpuInput) input_ptr,
    daxa_RWBufferPtr(GpuGlobals) globals_ptr) {
    const f32 mouse_sens = 1.0;

    PLAYER.rot.z += INPUT.mouse.pos_delta.x * mouse_sens * SETTINGS.sensitivity * 0.001;
    // PLAYER.rot.y = fract(INPUT.time * 0.5) * 2.0 * PI;
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
    PLAYER.cam.prev_tan_half_fov = PLAYER.cam.tan_half_fov;
    PLAYER.cam.tan_half_fov = tan(SETTINGS.fov * 0.5);

    f32vec3 move_vec = f32vec3(0, 0, 0);
    PLAYER.forward = f32vec3(+sin_rot_z, +cos_rot_z, 0);
    PLAYER.lateral = f32vec3(+cos_rot_z, -sin_rot_z, 0);

    const f32 accel_rate = 30.0;
    const f32 speed = 2.5;
    const f32 sprint_speed = 20.5;

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

#if ENABLE_CHUNK_WRAPPING
    const u32vec3 chunk_n = u32vec3(1u << SETTINGS.log2_chunks_per_axis);
    const f32vec3 HALF_CHUNK_N = f32vec3(chunk_n) * 0.5;
    PLAYER.chunk_offset += i32vec3(floor(PLAYER.pos / CHUNK_WORLDSPACE_SIZE - HALF_CHUNK_N + 0.5));
    PLAYER.pos = mod(PLAYER.pos - 0.5 * CHUNK_WORLDSPACE_SIZE, CHUNK_WORLDSPACE_SIZE) + CHUNK_WORLDSPACE_SIZE * (HALF_CHUNK_N - 0.5);
#else
    // Logic to recover when debugging, and toggling the ENABLE_CHUNK_WRAPPING define!
    PLAYER.pos += f32vec3(PLAYER.chunk_offset) * CHUNK_WORLDSPACE_SIZE;
    PLAYER.chunk_offset = i32vec3(0);
#endif

    PLAYER.cam.pos = PLAYER.pos + f32vec3(0, 0, 0);

    float aspect = float(INPUT.frame_dim.x) / float(INPUT.frame_dim.y);
    float near = 0.01;

    vec3 eye = PLAYER.cam.pos;
    vec3 center = PLAYER.cam.pos + PLAYER.cam.rot_mat * vec3(0, 0, 1);
    vec3 up = vec3(0, 0, 1);

    vec3 f = normalize(center - eye);
    vec3 s = normalize(cross(f, up));
    vec3 u = cross(s, f);

    mat4 proj_mat = mat4(0.0);
    proj_mat[0][0] = +1.0 / PLAYER.cam.tan_half_fov / aspect;
    proj_mat[1][1] = -1.0 / PLAYER.cam.tan_half_fov;
    proj_mat[2][2] = +0.0;
    proj_mat[2][3] = -1.0;
    proj_mat[3][2] = near;

    mat4 view_mat = mat4(0.0);
    view_mat[0][0] = s.x;
    view_mat[1][0] = s.y;
    view_mat[2][0] = s.z;
    view_mat[0][1] = u.x;
    view_mat[1][1] = u.y;
    view_mat[2][1] = u.z;
    view_mat[0][2] = -f.x;
    view_mat[1][2] = -f.y;
    view_mat[2][2] = -f.z;
    view_mat[3][0] = -dot(s, eye);
    view_mat[3][1] = -dot(u, eye);
    view_mat[3][2] = dot(f, eye);
    view_mat[3][3] = 1.0;

    PLAYER.cam.proj_mat = proj_mat * view_mat;
}
#undef PLAYER
#undef INPUT
#undef SETTINGS

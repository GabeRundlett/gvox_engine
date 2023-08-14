#pragma once

#include <shared/app.inl>

#include <utils/math.glsl>

#define PLAYER deref(globals_ptr).player
void player_fix_chunk_offset(
    daxa_BufferPtr(GpuInput) input_ptr,
    daxa_RWBufferPtr(GpuGlobals) globals_ptr) {
#if ENABLE_CHUNK_WRAPPING
    const u32vec3 chunk_n = u32vec3(1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);
    const f32vec3 HALF_CHUNK_N = f32vec3(chunk_n) * 0.5;
    PLAYER.chunk_offset += i32vec3(floor(PLAYER.pos / CHUNK_WORLDSPACE_SIZE - HALF_CHUNK_N + 0.5));
    PLAYER.pos = mod(PLAYER.pos - 0.5 * CHUNK_WORLDSPACE_SIZE, CHUNK_WORLDSPACE_SIZE) + CHUNK_WORLDSPACE_SIZE * (HALF_CHUNK_N - 0.5);
#else
    // Logic to recover when debugging, and toggling the ENABLE_CHUNK_WRAPPING define!
    PLAYER.pos += f32vec3(PLAYER.chunk_offset) * CHUNK_WORLDSPACE_SIZE;
    PLAYER.chunk_offset = i32vec3(0);
#endif
}
#undef PLAYER

#define PLAYER deref(globals_ptr).player
void player_startup(
    daxa_BufferPtr(GpuInput) input_ptr,
    daxa_RWBufferPtr(GpuGlobals) globals_ptr) {
    PLAYER.pos = f32vec3(0.01, 0.02, 0.03);
    // PLAYER.pos = f32vec3(150.01, 150.02, 80.03);
    // PLAYER.pos = f32vec3(66.01, 38.02, 14.01);

    // Inside beach hut
    // PLAYER.pos = f32vec3(173.78, 113.72, 12.09);

    PLAYER.pitch = PI * 0.349;
    PLAYER.yaw = PI * 0.25;

    // PLAYER.pitch = PI * 0.249;
    // PLAYER.yaw = PI * 1.25;

    player_fix_chunk_offset(input_ptr, globals_ptr);
}
#undef PLAYER

#define INPUT deref(input_ptr)
#define PLAYER deref(globals_ptr).player
void player_perframe(
    daxa_BufferPtr(GpuInput) input_ptr,
    daxa_RWBufferPtr(GpuGlobals) globals_ptr) {
    const f32 mouse_sens = 1.0;

    if (INPUT.actions[GAME_ACTION_INTERACT1] != 0) {
        PLAYER.roll += INPUT.mouse.pos_delta.x * mouse_sens * INPUT.sensitivity * 0.001;
    } else {
        PLAYER.yaw += INPUT.mouse.pos_delta.x * mouse_sens * INPUT.sensitivity * 0.001;
        PLAYER.pitch -= INPUT.mouse.pos_delta.y * mouse_sens * INPUT.sensitivity * 0.001;
    }

    const float MAX_ROT_EPS = 0.01;
    PLAYER.pitch = clamp(PLAYER.pitch, MAX_ROT_EPS, PI - MAX_ROT_EPS);
    float sin_rot_x = sin(PLAYER.pitch), cos_rot_x = cos(PLAYER.pitch);
    float sin_rot_z = sin(PLAYER.yaw), cos_rot_z = cos(PLAYER.yaw);

    f32vec3 move_vec = f32vec3(0, 0, 0);
    PLAYER.forward = f32vec3(+sin_rot_z, +cos_rot_z, 0);
    PLAYER.lateral = f32vec3(+cos_rot_z, -sin_rot_z, 0);

    const f32 accel_rate = 30.0;
    const f32 speed = 2.5;
    const f32 sprint_speed = 25.5;

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

    player_fix_chunk_offset(input_ptr, globals_ptr);

    float tan_half_fov = tan(INPUT.fov * 0.5);
    float aspect = float(INPUT.frame_dim.x) / float(INPUT.frame_dim.y);
    float near = 0.01;

    PLAYER.cam.prev_view_to_prev_clip = PLAYER.cam.view_to_clip;
    PLAYER.cam.prev_clip_to_prev_view = PLAYER.cam.clip_to_view;
    PLAYER.cam.prev_world_to_prev_view = PLAYER.cam.world_to_view;
    PLAYER.cam.prev_view_to_prev_world = PLAYER.cam.view_to_world;

    memoryBarrier();

    PLAYER.cam.view_to_clip = f32mat4x4(0.0);
    PLAYER.cam.view_to_clip[0][0] = +1.0 / tan_half_fov / aspect;
    PLAYER.cam.view_to_clip[1][1] = +1.0 / tan_half_fov;
    PLAYER.cam.view_to_clip[2][2] = +0.0;
    PLAYER.cam.view_to_clip[2][3] = -1.0;
    PLAYER.cam.view_to_clip[3][2] = near;

    PLAYER.cam.clip_to_view = f32mat4x4(0.0);
    PLAYER.cam.clip_to_view[0][0] = tan_half_fov * aspect;
    PLAYER.cam.clip_to_view[1][1] = tan_half_fov;
    PLAYER.cam.clip_to_view[2][2] = +0.0;
    PLAYER.cam.clip_to_view[2][3] = +1.0 / near;
    PLAYER.cam.clip_to_view[3][2] = -1.0;

    f32vec2 sample_offset = f32vec2(
        INPUT.halton_jitter.x / float(INPUT.frame_dim.x),
        INPUT.halton_jitter.y / float(INPUT.frame_dim.y));

    f32vec4 output_tex_size = f32vec4(deref(input_ptr).frame_dim.xy, 0, 0);
    output_tex_size.zw = 1.0 / output_tex_size.xy;

    f32mat4x4 jitter_mat = f32mat4x4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        uv_to_ss(input_ptr, f32vec2(0.0), output_tex_size), 0, 1);
    f32mat4x4 inv_jitter_mat = f32mat4x4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        ss_to_uv(input_ptr, f32vec2(0.0), output_tex_size), 0, 1);

    PLAYER.cam.view_to_sample = jitter_mat * PLAYER.cam.view_to_clip;
    PLAYER.cam.sample_to_view = PLAYER.cam.clip_to_view * inv_jitter_mat;

    PLAYER.cam.view_to_world = translation_matrix(PLAYER.pos) * rotation_matrix(PLAYER.yaw, PLAYER.pitch, PLAYER.roll);
    PLAYER.cam.world_to_view = inv_rotation_matrix(PLAYER.yaw, PLAYER.pitch, PLAYER.roll) * translation_matrix(-PLAYER.pos);

    PLAYER.cam.clip_to_prev_clip =
        PLAYER.cam.prev_view_to_prev_clip *
        PLAYER.cam.prev_world_to_prev_view *
        PLAYER.cam.view_to_world *
        PLAYER.cam.clip_to_view;
}
#undef PLAYER
#undef INPUT

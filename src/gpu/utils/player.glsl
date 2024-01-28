#pragma once

#include <shared/app.inl>

#include <utils/math.glsl>

#define PLAYER deref(globals_ptr).player
void player_fix_chunk_offset(
    daxa_BufferPtr(GpuInput) input_ptr,
    daxa_RWBufferPtr(GpuGlobals) globals_ptr) {
    PLAYER.prev_unit_offset = PLAYER.player_unit_offset;
#if ENABLE_CHUNK_WRAPPING
    PLAYER.player_unit_offset += daxa_i32vec3(floor(PLAYER.pos));
    PLAYER.pos = fract(PLAYER.pos);
#else
    // Logic to recover when debugging, and toggling the ENABLE_CHUNK_WRAPPING define!
    PLAYER.pos += daxa_f32vec3(PLAYER.player_unit_offset);
    PLAYER.player_unit_offset = daxa_i32vec3(0);
#endif
}
#undef PLAYER

#define PLAYER deref(globals_ptr).player
void player_startup(
    daxa_BufferPtr(GpuInput) input_ptr,
    daxa_RWBufferPtr(GpuGlobals) globals_ptr) {
    PLAYER.pos = daxa_f32vec3(0.01, 0.02, 0.03);
    PLAYER.vel = daxa_f32vec3(0.0);
    // PLAYER.pos = daxa_f32vec3(150.01, 150.02, 80.03);
    // PLAYER.pos = daxa_f32vec3(66.01, 38.02, 14.01);

    // Inside beach hut
    // PLAYER.pos = daxa_f32vec3(173.78, 113.72, 12.09);

    PLAYER.pitch = M_PI * 0.349;
    PLAYER.yaw = M_PI * 0.25;

    // PLAYER.pitch = M_PI * 0.249;
    // PLAYER.yaw = M_PI * 1.25;

    // Inside Bistro
    // PLAYER.pos = daxa_f32vec3(22.63, 51.60, 43.82);
    // PLAYER.yaw = 1.68;
    // PLAYER.pitch = 1.49;

    player_fix_chunk_offset(input_ptr, globals_ptr);
}
#undef PLAYER

#define INPUT deref(input_ptr)
#define PLAYER deref(globals_ptr).player
void player_perframe(
    daxa_BufferPtr(GpuInput) input_ptr,
    daxa_RWBufferPtr(GpuGlobals) globals_ptr) {
    const daxa_f32 mouse_sens = 1.0;

    if (INPUT.actions[GAME_ACTION_INTERACT1] != 0) {
        PLAYER.roll += INPUT.mouse.pos_delta.x * mouse_sens * INPUT.sensitivity * 0.001;
    } else {
        PLAYER.yaw += INPUT.mouse.pos_delta.x * mouse_sens * INPUT.sensitivity * 0.001;
        PLAYER.pitch -= INPUT.mouse.pos_delta.y * mouse_sens * INPUT.sensitivity * 0.001;
    }

    const float MAX_ROT_EPS = 0.01;
    PLAYER.pitch = clamp(PLAYER.pitch, MAX_ROT_EPS, M_PI - MAX_ROT_EPS);
    float sin_rot_x = sin(PLAYER.pitch), cos_rot_x = cos(PLAYER.pitch);
    float sin_rot_z = sin(PLAYER.yaw), cos_rot_z = cos(PLAYER.yaw);

    daxa_f32vec3 move_vec = daxa_f32vec3(0, 0, 0);
    PLAYER.forward = daxa_f32vec3(+sin_rot_z, +cos_rot_z, 0);
    PLAYER.lateral = daxa_f32vec3(+cos_rot_z, -sin_rot_z, 0);

    const bool is_flying = true;

    const daxa_f32 accel_rate = 30.0;
    const daxa_f32 speed = 2.5;
    const daxa_f32 sprint_speed = is_flying ? 25.5 : 2.5;

    const daxa_f32 MAX_SPEED = speed * sprint_speed;

    if (INPUT.actions[GAME_ACTION_MOVE_FORWARD] != 0)
        move_vec += PLAYER.forward;
    if (INPUT.actions[GAME_ACTION_MOVE_BACKWARD] != 0)
        move_vec -= PLAYER.forward;
    if (INPUT.actions[GAME_ACTION_MOVE_LEFT] != 0)
        move_vec -= PLAYER.lateral;
    if (INPUT.actions[GAME_ACTION_MOVE_RIGHT] != 0)
        move_vec += PLAYER.lateral;

    daxa_f32 applied_speed = speed;
    if ((INPUT.actions[GAME_ACTION_SPRINT] != 0) == is_flying)
        applied_speed *= sprint_speed;

    if (is_flying) {
        if (INPUT.actions[GAME_ACTION_JUMP] != 0)
            move_vec += daxa_f32vec3(0, 0, 1);
        if (INPUT.actions[GAME_ACTION_CROUCH] != 0)
            move_vec -= daxa_f32vec3(0, 0, 1);

        PLAYER.vel = move_vec * applied_speed;
        PLAYER.pos += PLAYER.vel * INPUT.delta_time;
    } else {
        vec3 pos = PLAYER.pos + daxa_f32vec3(PLAYER.player_unit_offset);
        vec3 vel = PLAYER.vel;

        daxa_f32vec3 nonvertical_vel = daxa_f32vec3(vel.xy, 0);

        vel += daxa_f32vec3(0, 0, -9.8) * INPUT.delta_time;
        pos += vel * INPUT.delta_time;

        bool is_on_ground = pos.z < 0.0;

        if (is_on_ground) {
            pos.z = 0.0;
            vel.z = 0.0;
            if (INPUT.actions[GAME_ACTION_JUMP] != 0)
                vel += daxa_f32vec3(0, 0, 3.0);

            apply_friction(input_ptr, vel, nonvertical_vel, 8.0);

            if (dot(nonvertical_vel, nonvertical_vel) > MAX_SPEED * MAX_SPEED) {
                vel.xy -= normalize(vel.xy) * MAX_SPEED;
            }
        } else {
        }

        if (dot(move_vec, move_vec) != 0.0) {
            move_vec = normalize(move_vec);
            vel += move_vec * max(applied_speed - dot(nonvertical_vel, move_vec), 0.0);
        }

        PLAYER.pos = pos - daxa_f32vec3(PLAYER.player_unit_offset);
        PLAYER.vel = vel;
    }

    player_fix_chunk_offset(input_ptr, globals_ptr);

    float tan_half_fov = tan(INPUT.fov * 0.5);
    float aspect = float(INPUT.frame_dim.x) / float(INPUT.frame_dim.y);
    float near = 0.01;

    PLAYER.cam.prev_view_to_prev_clip = PLAYER.cam.view_to_clip;
    PLAYER.cam.prev_clip_to_prev_view = PLAYER.cam.clip_to_view;
    PLAYER.cam.prev_world_to_prev_view = PLAYER.cam.world_to_view;
    PLAYER.cam.prev_view_to_prev_world = PLAYER.cam.view_to_world;

    memoryBarrier();

    PLAYER.cam.view_to_clip = daxa_f32mat4x4(0.0);
    PLAYER.cam.view_to_clip[0][0] = +1.0 / tan_half_fov / aspect;
    PLAYER.cam.view_to_clip[1][1] = +1.0 / tan_half_fov;
    PLAYER.cam.view_to_clip[2][2] = +0.0;
    PLAYER.cam.view_to_clip[2][3] = -1.0;
    PLAYER.cam.view_to_clip[3][2] = near;

    PLAYER.cam.clip_to_view = daxa_f32mat4x4(0.0);
    PLAYER.cam.clip_to_view[0][0] = tan_half_fov * aspect;
    PLAYER.cam.clip_to_view[1][1] = tan_half_fov;
    PLAYER.cam.clip_to_view[2][2] = +0.0;
    PLAYER.cam.clip_to_view[2][3] = +1.0 / near;
    PLAYER.cam.clip_to_view[3][2] = -1.0;

    daxa_f32vec2 sample_offset = daxa_f32vec2(
        INPUT.halton_jitter.x / float(INPUT.frame_dim.x),
        INPUT.halton_jitter.y / float(INPUT.frame_dim.y));

    daxa_f32vec4 output_tex_size = daxa_f32vec4(deref(input_ptr).frame_dim.xy, 0, 0);
    output_tex_size.zw = 1.0 / output_tex_size.xy;

    daxa_f32mat4x4 jitter_mat = daxa_f32mat4x4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        uv_to_ss(input_ptr, daxa_f32vec2(0.0), output_tex_size), 0, 1);
    daxa_f32mat4x4 inv_jitter_mat = daxa_f32mat4x4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        ss_to_uv(input_ptr, daxa_f32vec2(0.0), output_tex_size), 0, 1);

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

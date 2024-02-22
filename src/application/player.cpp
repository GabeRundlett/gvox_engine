#include "player.hpp"

#include <bit>
#include <glm/glm.hpp>
#include <fmt/format.h>

#include <application/settings.hpp>
#include <renderer/kajiya/inc/math_const.glsl>
#include <utilities/debug.hpp>

using vec4 = daxa_f32vec4;
using vec2 = daxa_f32vec2;
using ivec3 = daxa_i32vec3;

float dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float length(vec3 v) {
    return sqrt(dot(v, v));
}

vec3 normalize(vec3 v) {
    float len = length(v);
    v.x /= len;
    v.y /= len;
    v.z /= len;
    return v;
}

vec3 operator+(vec3 a, vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
ivec3 operator+(ivec3 a, ivec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
vec3 operator-(vec3 a, vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
vec3 operator*(vec3 a, vec3 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}
vec3 operator*(vec3 a, float b) {
    return {a.x * b, a.y * b, a.z * b};
}

using std::clamp;

glm::mat4 rotation_matrix(float yaw, float pitch, float roll) {
    float sin_rot_x = sin(pitch), cos_rot_x = cos(pitch);
    float sin_rot_y = sin(roll), cos_rot_y = cos(roll);
    float sin_rot_z = sin(yaw), cos_rot_z = cos(yaw);
    return glm::mat4(
               cos_rot_z, -sin_rot_z, 0, 0,
               sin_rot_z, cos_rot_z, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1) *
           glm::mat4(
               1, 0, 0, 0,
               0, cos_rot_x, sin_rot_x, 0,
               0, -sin_rot_x, cos_rot_x, 0,
               0, 0, 0, 1) *
           glm::mat4(
               cos_rot_y, -sin_rot_y, 0, 0,
               sin_rot_y, cos_rot_y, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1);
}
glm::mat4 inv_rotation_matrix(float yaw, float pitch, float roll) {
    float sin_rot_x = sin(-pitch), cos_rot_x = cos(-pitch);
    float sin_rot_y = sin(-roll), cos_rot_y = cos(-roll);
    float sin_rot_z = sin(-yaw), cos_rot_z = cos(-yaw);
    return glm::mat4(
               cos_rot_y, -sin_rot_y, 0, 0,
               sin_rot_y, cos_rot_y, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1) *
           glm::mat4(
               1, 0, 0, 0,
               0, cos_rot_x, sin_rot_x, 0,
               0, -sin_rot_x, cos_rot_x, 0,
               0, 0, 0, 1) *
           glm::mat4(
               cos_rot_z, -sin_rot_z, 0, 0,
               sin_rot_z, cos_rot_z, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1);
}
glm::mat4 translation_matrix(vec3 pos) {
    return glm::mat4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        pos.x, pos.y, pos.z, 1);
}

vec3 apply_inv_rotation(vec3 pt, vec3 ypr) {
    float yaw = ypr.x;
    float pitch = ypr.y;
    float roll = ypr.z;
    auto res = inv_rotation_matrix(yaw, pitch, roll) * glm::vec4(pt.x, pt.y, pt.z, 0.0);
    return {res.x, res.y, res.z};
}

void player_fix_chunk_offset(Player &PLAYER) {
    PLAYER.prev_unit_offset = PLAYER.player_unit_offset;
#if ENABLE_CHUNK_WRAPPING
    const bool wrap_position = AppSettings::get<settings::Checkbox>("Player", "Wrap Position").value;
    if (wrap_position) {
        PLAYER.player_unit_offset = PLAYER.player_unit_offset + ivec3(floor(PLAYER.pos.x), floor(PLAYER.pos.y), floor(PLAYER.pos.z));
        PLAYER.pos = {PLAYER.pos.x - floor(PLAYER.pos.x), PLAYER.pos.y - floor(PLAYER.pos.y), PLAYER.pos.z - floor(PLAYER.pos.z)};
    }
#else
    // Logic to recover when debugging, and toggling the ENABLE_CHUNK_WRAPPING define!
    PLAYER.pos = PLAYER.pos + vec3(PLAYER.player_unit_offset.x, PLAYER.player_unit_offset.y, PLAYER.player_unit_offset.z);
    PLAYER.player_unit_offset = ivec3(0);
#endif
}

void player_startup(Player &PLAYER) {
    AppSettings::add<settings::InputFloat>({"Player", "Movement Speed", {.value = 2.5f}});
    AppSettings::add<settings::InputFloat>({"Player", "Sprint Multiplier", {.value = 25.0f}});
    AppSettings::add<settings::Checkbox>({"Player", "Wrap Position", {.value = true}});

    PLAYER.pos = vec3(0.01f, 0.02f, 0.03f);
    PLAYER.vel = vec3(0.0);
    PLAYER.player_unit_offset = ivec3(0, 0, 0);
    // PLAYER.pos = vec3(150.01, 150.02, 80.03);
    // PLAYER.pos = vec3(66.01, 38.02, 14.01);

    // Inside beach hut
    // PLAYER.pos = vec3(173.78f - 125, 113.72f - 125, 12.09f);

    PLAYER.pitch = float(M_PI * 0.349);
    PLAYER.yaw = float(M_PI * 0.25);

    // PLAYER.pitch = M_PI * 0.249;
    // PLAYER.yaw = M_PI * 1.25;

    PLAYER.roll = 0.0f;

    // Inside Bistro
    // PLAYER.pos = vec3(22.63, 51.60, 43.82);
    // PLAYER.yaw = 1.68;
    // PLAYER.pitch = 1.49;

    player_fix_chunk_offset(PLAYER);
}

void player_perframe(PlayerInput &INPUT, Player &PLAYER) {
    const float mouse_sens = 1.0f;

    if (INPUT.actions[GAME_ACTION_INTERACT1] != 0) {
        PLAYER.roll += INPUT.mouse.pos_delta.x * mouse_sens * INPUT.sensitivity * 0.001f;
    } else {
        PLAYER.yaw += INPUT.mouse.pos_delta.x * mouse_sens * INPUT.sensitivity * 0.001f;
        PLAYER.pitch -= INPUT.mouse.pos_delta.y * mouse_sens * INPUT.sensitivity * 0.001f;
    }

    const float MAX_ROT_EPS = 0.01f;
    PLAYER.pitch = clamp(PLAYER.pitch, MAX_ROT_EPS, float(M_PI) - MAX_ROT_EPS);
    // float sin_rot_x = sinf(PLAYER.pitch), cos_rot_x = cosf(PLAYER.pitch);
    float sin_rot_z = sinf(PLAYER.yaw), cos_rot_z = cosf(PLAYER.yaw);

    vec3 move_vec = vec3(0, 0, 0);
    PLAYER.forward = vec3(+sin_rot_z, +cos_rot_z, 0);
    PLAYER.lateral = vec3(+cos_rot_z, -sin_rot_z, 0);

    const bool is_flying = true;

    const float speed = AppSettings::get<settings::InputFloat>("Player", "Movement Speed").value;
    const float sprint_speed = AppSettings::get<settings::InputFloat>("Player", "Sprint Multiplier").value;

    if (INPUT.actions[GAME_ACTION_MOVE_FORWARD] != 0)
        move_vec = move_vec + PLAYER.forward;
    if (INPUT.actions[GAME_ACTION_MOVE_BACKWARD] != 0)
        move_vec = move_vec - PLAYER.forward;
    if (INPUT.actions[GAME_ACTION_MOVE_LEFT] != 0)
        move_vec = move_vec - PLAYER.lateral;
    if (INPUT.actions[GAME_ACTION_MOVE_RIGHT] != 0)
        move_vec = move_vec + PLAYER.lateral;

    float applied_speed = speed;
    if ((INPUT.actions[GAME_ACTION_SPRINT] != 0) == is_flying)
        applied_speed *= sprint_speed;

    if (INPUT.actions[GAME_ACTION_JUMP] != 0)
        move_vec = move_vec + vec3(0, 0, 1);
    if (INPUT.actions[GAME_ACTION_CROUCH] != 0)
        move_vec = move_vec - vec3(0, 0, 1);

    PLAYER.vel = move_vec * applied_speed;
    PLAYER.pos = PLAYER.pos + PLAYER.vel * INPUT.delta_time;

    player_fix_chunk_offset(PLAYER);

    float tan_half_fov = tan(INPUT.fov * 0.5f);
    float aspect = float(INPUT.frame_dim.x) / float(INPUT.frame_dim.y);
    float near = 0.01f;

    PLAYER.cam.prev_view_to_prev_clip = PLAYER.cam.view_to_clip;
    PLAYER.cam.prev_clip_to_prev_view = PLAYER.cam.clip_to_view;
    PLAYER.cam.prev_world_to_prev_view = PLAYER.cam.world_to_view;
    PLAYER.cam.prev_view_to_prev_world = PLAYER.cam.view_to_world;

    PLAYER.cam.view_to_clip = mat4{};
    PLAYER.cam.view_to_clip.x.x = +1.0f / tan_half_fov / aspect;
    PLAYER.cam.view_to_clip.y.y = +1.0f / tan_half_fov;
    PLAYER.cam.view_to_clip.z.z = +0.0f;
    PLAYER.cam.view_to_clip.z.w = -1.0f;
    PLAYER.cam.view_to_clip.w.z = near;

    PLAYER.cam.clip_to_view = mat4{};
    PLAYER.cam.clip_to_view.x.x = tan_half_fov * aspect;
    PLAYER.cam.clip_to_view.y.y = tan_half_fov;
    PLAYER.cam.clip_to_view.z.z = +0.0f;
    PLAYER.cam.clip_to_view.z.w = +1.0f / near;
    PLAYER.cam.clip_to_view.w.z = -1.0f;

    vec2 sample_offset = vec2(
        INPUT.halton_jitter.x / float(INPUT.frame_dim.x),
        INPUT.halton_jitter.y / float(INPUT.frame_dim.y));

    glm::mat4 clip_to_sample = glm::mat4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        sample_offset.x * -2.0f, sample_offset.y * -2.0f, 0, 1);

    glm::mat4 sample_to_clip = glm::mat4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        sample_offset.x * +2.0f, sample_offset.y * +2.0f, 0, 1);

    PLAYER.cam.view_to_sample = std::bit_cast<mat4>(clip_to_sample * std::bit_cast<glm::mat4>(PLAYER.cam.view_to_clip));
    PLAYER.cam.sample_to_view = std::bit_cast<mat4>(std::bit_cast<glm::mat4>(PLAYER.cam.clip_to_view) * sample_to_clip);

    PLAYER.cam.view_to_world = std::bit_cast<mat4>(translation_matrix(PLAYER.pos) * rotation_matrix(PLAYER.yaw, PLAYER.pitch, PLAYER.roll));
    PLAYER.cam.world_to_view = std::bit_cast<mat4>(inv_rotation_matrix(PLAYER.yaw, PLAYER.pitch, PLAYER.roll) * translation_matrix(PLAYER.pos * -1.0f));

    PLAYER.cam.clip_to_prev_clip = std::bit_cast<mat4>(
        std::bit_cast<glm::mat4>(PLAYER.cam.prev_view_to_prev_clip) *
        std::bit_cast<glm::mat4>(PLAYER.cam.prev_world_to_prev_view) *
        std::bit_cast<glm::mat4>(PLAYER.cam.view_to_world) *
        std::bit_cast<glm::mat4>(PLAYER.cam.clip_to_view));

    debug_utils::DebugDisplay::set_debug_string("Player Pos", fmt::format("{:.3f}, {:.3f}, {:.3f}", PLAYER.pos.x, PLAYER.pos.y, PLAYER.pos.z));
    debug_utils::DebugDisplay::set_debug_string("Player Rot (Y/P/R)", fmt::format("{:.3f}, {:.3f}, {:.3f}", PLAYER.yaw, PLAYER.pitch, PLAYER.roll));
    debug_utils::DebugDisplay::set_debug_string("Player Unit Offset", fmt::format("{}, {}, {}", PLAYER.player_unit_offset.x, PLAYER.player_unit_offset.y, PLAYER.player_unit_offset.z));
}

#include "player.hpp"

#include <bit>
#include <fmt/format.h>

#include <application/settings.hpp>
#include <utilities/debug.hpp>
#include <utilities/math.hpp>
using std::clamp;

void player_fix_chunk_offset(Player &PLAYER) {
    PLAYER.prev_unit_offset = PLAYER.player_unit_offset;
#if ENABLE_CHUNK_WRAPPING
    const bool wrap_position = AppSettings::get<settings::Checkbox>("Player", "Wrap Position").value;
    if (wrap_position) {
        PLAYER.player_unit_offset = PLAYER.player_unit_offset + daxa_i32vec3(floor(PLAYER.pos.x), floor(PLAYER.pos.y), floor(PLAYER.pos.z));
        PLAYER.pos = {PLAYER.pos.x - floor(PLAYER.pos.x), PLAYER.pos.y - floor(PLAYER.pos.y), PLAYER.pos.z - floor(PLAYER.pos.z)};
    }
#else
    // Logic to recover when debugging, and toggling the ENABLE_CHUNK_WRAPPING define!
    PLAYER.pos = PLAYER.pos + daxa_f32vec3(PLAYER.player_unit_offset.x, PLAYER.player_unit_offset.y, PLAYER.player_unit_offset.z);
    PLAYER.player_unit_offset = daxa_i32vec3(0);
#endif
}

void player_startup(Player &PLAYER) {
    if (((PLAYER.flags >> 0) & 0x1) != 0) {
        return;
    }
    PLAYER.flags = 1;

    AppSettings::add<settings::InputFloat>({"Player", "Movement Speed", {.value = 2.5f}});
    AppSettings::add<settings::InputFloat>({"Player", "Sprint Multiplier", {.value = 3.0f}});
    AppSettings::add<settings::InputFloat>({"Player", "Crouch Multiplier", {.value = 0.5f}});
    AppSettings::add<settings::Checkbox>({"Player", "Wrap Position", {.value = true}});
    AppSettings::add<settings::InputFloat>({"Player", "Jump Strength (meters on Earth)", {.value = 2.0f}});
    AppSettings::add<settings::InputFloat>({"Player", "Crouch Height", {.value = 1.0f}});

    // float ground_level = AppSettings::get<settings::InputFloat>("Atmosphere", "atmosphere_bottom").value * 1000.0f + 2000.0f;
    float ground_level = 0.0f;

    PLAYER.pos = vec3(0.01f, 0.02f, 0.03f + ground_level);
    PLAYER.cam_pos_offset = vec3(0.0);
    PLAYER.vel = vec3(0.0);
    PLAYER.player_unit_offset = daxa_i32vec3(0, 0, 0);
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

void toggle_fly(Player &PLAYER) {
    auto is_flying = (PLAYER.flags >> 6) & 0x1;
    auto toggled_last_frame = (PLAYER.flags >> 5) & 0x1;
    if (!toggled_last_frame) {
        PLAYER.flags = (PLAYER.flags & ~(0x1 << 6)) | ((1 - is_flying) << 6);
    }
    PLAYER.flags |= (0x1 << 5);
}

#define EARTH_GRAV 9.807f
#define MOON_GRAV 1.625f
#define MARS_GRAV 3.728f
#define JUPITER_GRAV 25.93f

#define GRAVITY EARTH_GRAV

#define EARTH_JUMP_HEIGHT 0.59

void player_perframe(PlayerInput &INPUT, Player &PLAYER, VoxelWorld &voxel_world) {
    const float mouse_sens = 1.0f;

    if (INPUT.actions[GAME_ACTION_INTERACT1] != 0) {
        PLAYER.roll += INPUT.mouse.pos_delta.x * mouse_sens * INPUT.sensitivity * 0.001f;
    } else {
        PLAYER.yaw += INPUT.mouse.pos_delta.x * mouse_sens * INPUT.sensitivity * 0.001f;
        PLAYER.pitch -= INPUT.mouse.pos_delta.y * mouse_sens * INPUT.sensitivity * 0.001f;
    }

    const float MAX_ROT_EPS = 0.0001f;
    PLAYER.pitch = clamp(PLAYER.pitch, MAX_ROT_EPS, float(M_PI) - MAX_ROT_EPS);
    // float sin_rot_x = sinf(PLAYER.pitch), cos_rot_x = cosf(PLAYER.pitch);
    float sin_rot_z = sinf(PLAYER.yaw), cos_rot_z = cosf(PLAYER.yaw);

    vec3 move_vec = vec3(0, 0, 0);
    auto move_forward = vec3(+sin_rot_z, +cos_rot_z, 0);
    auto move_lateral = vec3(+cos_rot_z, -sin_rot_z, 0);

    auto view_to_world = std::bit_cast<glm::mat4>(PLAYER.cam.view_to_world);
    auto forward_h = view_to_world * glm::vec4(0, 0, -1, 0);
    auto forward = glm::normalize(glm::vec3(forward_h.x, forward_h.y, forward_h.z));
    auto lateral_h = view_to_world * glm::vec4(+1, 0, 0, 0);
    auto lateral = glm::normalize(glm::vec3(lateral_h.x, lateral_h.y, lateral_h.z));

    PLAYER.forward = std::bit_cast<vec3>(forward);
    PLAYER.lateral = std::bit_cast<vec3>(lateral);

    if (INPUT.actions[GAME_ACTION_TOGGLE_FLY] != 0) {
        toggle_fly(PLAYER);
        PLAYER.vel = vec3(0, 0, 0);
    } else {
        PLAYER.flags &= ~(1u << 5);
    }

    const bool is_flying = ((PLAYER.flags >> 6) & 0x1) == 1;
    const bool is_on_ground = ((PLAYER.flags >> 1) & 0x1) == 1;
    const bool is_crouched = ((PLAYER.flags >> 2) & 0x1) == 1;

    const float speed = AppSettings::get<settings::InputFloat>("Player", "Movement Speed").value;
    const float sprint_speed = AppSettings::get<settings::InputFloat>("Player", "Sprint Multiplier").value;
    const float crouch_speed = AppSettings::get<settings::InputFloat>("Player", "Crouch Multiplier").value;
    const float jump_strength = AppSettings::get<settings::InputFloat>("Player", "Jump Strength (meters on Earth)").value;
    const float crouch_height = AppSettings::get<settings::InputFloat>("Player", "Crouch Height").value;

    if (INPUT.actions[GAME_ACTION_MOVE_FORWARD] != 0)
        move_vec = move_vec + move_forward;
    if (INPUT.actions[GAME_ACTION_MOVE_BACKWARD] != 0)
        move_vec = move_vec - move_forward;
    if (INPUT.actions[GAME_ACTION_MOVE_LEFT] != 0)
        move_vec = move_vec - move_lateral;
    if (INPUT.actions[GAME_ACTION_MOVE_RIGHT] != 0)
        move_vec = move_vec + move_lateral;

    float applied_speed = speed;
    if ((INPUT.actions[GAME_ACTION_SPRINT] != 0) != 0 && !is_crouched)
        applied_speed *= sprint_speed;
    if (is_crouched)
        applied_speed *= crouch_speed;

    vec3 acc = vec3(0, 0, 0);

    float player_height = 1.75f;

    if (is_flying) {
        if (INPUT.actions[GAME_ACTION_JUMP] != 0)
            move_vec = move_vec + vec3(0, 0, 1);
        if (INPUT.actions[GAME_ACTION_CROUCH] != 0)
            move_vec = move_vec - vec3(0, 0, 1);
    } else {
        if (is_on_ground && INPUT.actions[GAME_ACTION_JUMP] != 0)
            PLAYER.vel.z = EARTH_GRAV * sqrt(jump_strength * 2.0 / EARTH_GRAV);
        else
            acc.z = -GRAVITY;

        if (INPUT.actions[GAME_ACTION_CROUCH] != 0) {
            if (!is_crouched) {
                PLAYER.pos.z -= player_height - crouch_height;
                PLAYER.cam_pos_offset.z += player_height - crouch_height;
            }
            player_height = crouch_height;
            PLAYER.flags |= (1u << 2);
        } else {
            if (is_crouched) {
                PLAYER.pos.z += player_height - crouch_height;
                PLAYER.cam_pos_offset.z -= player_height - crouch_height;
            }
            PLAYER.flags &= ~(1u << 2);
        }
    }

    float dt = std::min(INPUT.delta_time, 1.0f);

    PLAYER.vel = PLAYER.vel + acc * dt;
    auto offset = (PLAYER.vel + move_vec * applied_speed) * dt;
    PLAYER.pos = PLAYER.pos + offset;

    PLAYER.flags &= ~(1u << 0x1);
    bool inside_terrain = false;
    int32_t voxel_height = player_height * VOXEL_SCL + 2;

    for (int32_t xi = -2; xi <= 2; ++xi) {
        for (int32_t yi = -2; yi <= 2; ++yi) {
            for (int32_t zi = 0; zi < voxel_height; ++zi) {
                auto in_voxel = voxel_world.sample(PLAYER.pos - vec3(0, 0, player_height) + vec3(xi * VOXEL_SIZE, yi * VOXEL_SIZE, zi * VOXEL_SIZE), PLAYER.player_unit_offset);
                if (in_voxel) {
                    inside_terrain = true;
                    break;
                }
            }
        }
    }

    if (inside_terrain) {
        bool space_above = false;
        int32_t first_height = -1;
        for (int32_t zi = 0; zi < voxel_height + voxel_height / 2; ++zi) {
            bool found_voxel = false;
            for (int32_t xi = -2; xi <= 2; ++xi) {
                if (found_voxel) {
                    break;
                }
                for (int32_t yi = -2; yi <= 2; ++yi) {
                    if (found_voxel) {
                        break;
                    }
                    auto solid = voxel_world.sample(PLAYER.pos - vec3(0, 0, player_height) + vec3(xi * VOXEL_SIZE, yi * VOXEL_SIZE, zi * VOXEL_SIZE), PLAYER.player_unit_offset);
                    if (solid) {
                        found_voxel = true;
                    }
                }
            }
            if (zi - first_height >= voxel_height) {
                break;
            }
            if (found_voxel) {
                first_height = -1;
                space_above = false;
            }
            if (!found_voxel && zi < voxel_height / 2 && first_height == -1) {
                first_height = zi;
                space_above = true;
            }
        }
        if (space_above) {
            float current_z = PLAYER.pos.z;
            PLAYER.pos = PLAYER.pos + vec3(0, 0, VOXEL_SIZE * first_height);
            PLAYER.pos.z = floor(PLAYER.pos.z * VOXEL_SCL) * VOXEL_SIZE;
            float new_z = PLAYER.pos.z;
            PLAYER.cam_pos_offset.z += current_z - new_z;
            PLAYER.flags |= (1u << 0x1);
            PLAYER.vel.z = 0;
        } else {
            PLAYER.pos = PLAYER.pos - offset;
        }
    }

    player_fix_chunk_offset(PLAYER);
    // debug_utils::DebugDisplay::set_debug_string("Player In Voxel", fmt::format("{}", in_voxel));

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

    daxa_f32vec2 sample_offset = daxa_f32vec2(
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

    vec3 cam_pos_offset_sign = sign(PLAYER.cam_pos_offset);
    const float interp_speed = std::max(length(PLAYER.cam_pos_offset) * VOXEL_SCL, 0.25f);
    PLAYER.cam_pos_offset = PLAYER.cam_pos_offset - cam_pos_offset_sign * dt * interp_speed;

    vec3 new_cam_pos_offset_sign = sign(PLAYER.cam_pos_offset);
    if (new_cam_pos_offset_sign.x != cam_pos_offset_sign.x)
        PLAYER.cam_pos_offset.x = 0.0f;
    if (new_cam_pos_offset_sign.y != cam_pos_offset_sign.y)
        PLAYER.cam_pos_offset.y = 0.0f;
    if (new_cam_pos_offset_sign.z != cam_pos_offset_sign.z)
        PLAYER.cam_pos_offset.z = 0.0f;

    auto cam_pos = PLAYER.pos + PLAYER.cam_pos_offset + vec3(0, 0, -0.1f);

    PLAYER.cam.view_to_sample = std::bit_cast<mat4>(clip_to_sample * std::bit_cast<glm::mat4>(PLAYER.cam.view_to_clip));
    PLAYER.cam.sample_to_view = std::bit_cast<mat4>(std::bit_cast<glm::mat4>(PLAYER.cam.clip_to_view) * sample_to_clip);

    PLAYER.cam.view_to_world = std::bit_cast<mat4>(translation_matrix(cam_pos) * rotation_matrix(PLAYER.yaw, PLAYER.pitch, PLAYER.roll));
    PLAYER.cam.world_to_view = std::bit_cast<mat4>(inv_rotation_matrix(PLAYER.yaw, PLAYER.pitch, PLAYER.roll) * translation_matrix(cam_pos * -1.0f));

    PLAYER.cam.clip_to_prev_clip = std::bit_cast<mat4>(
        std::bit_cast<glm::mat4>(PLAYER.cam.prev_view_to_prev_clip) *
        std::bit_cast<glm::mat4>(PLAYER.cam.prev_world_to_prev_view) *
        std::bit_cast<glm::mat4>(PLAYER.cam.view_to_world) *
        std::bit_cast<glm::mat4>(PLAYER.cam.clip_to_view));

    debug_utils::DebugDisplay::set_debug_string("Player Pos", fmt::format("{:.3f}, {:.3f}, {:.3f}", PLAYER.pos.x, PLAYER.pos.y, PLAYER.pos.z));
    debug_utils::DebugDisplay::set_debug_string("Player Pos (camera)", fmt::format("{:.3f}, {:.3f}, {:.3f}", cam_pos.x, cam_pos.y, cam_pos.z));
    debug_utils::DebugDisplay::set_debug_string("Player Pos (voxel)", fmt::format("{:.3f}, {:.3f}, {:.3f}", PLAYER.pos.x * VOXEL_SCL, PLAYER.pos.y * VOXEL_SCL, PLAYER.pos.z * VOXEL_SCL));
    debug_utils::DebugDisplay::set_debug_string("Player Rot (Y/P/R)", fmt::format("{:.3f}, {:.3f}, {:.3f}", PLAYER.yaw, PLAYER.pitch, PLAYER.roll));
    debug_utils::DebugDisplay::set_debug_string("Player Unit Offset", fmt::format("{}, {}, {}", PLAYER.player_unit_offset.x, PLAYER.player_unit_offset.y, PLAYER.player_unit_offset.z));
}

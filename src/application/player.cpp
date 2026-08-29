#include "player.hpp"

#include <bit>
#include <base/format.hpp>
#include <base/profiler.hpp>

#include <application/settings.hpp>
#include <utilities/debug.hpp>
#include <utilities/math.hpp>
#include <voxels/voxel_world.hpp>

using glm::clamp;

// Player flag bits (see also the is_* reads in player_perframe).
// Bits 0-3 hold the view mode, of which bit 0 doubles as "third person".
#define PLAYER_VIEW_MASK 0xfu
#define PLAYER_FLAG_THIRD_PERSON (1u << 0)
#define PLAYER_FLAG_VIEW_LATCH (1u << 4)
#define PLAYER_FLAG_FLY_LATCH (1u << 5)
#define PLAYER_FLAG_FLYING (1u << 6)
#define PLAYER_FLAG_CROUCHED (1u << 7)
#define PLAYER_FLAG_ON_GROUND (1u << 8)
#define PLAYER_FLAG_NEEDS_SPAWN (1u << 9)
#define PLAYER_COYOTE_SHIFT 10
#define PLAYER_COYOTE_MASK (0x7u << PLAYER_COYOTE_SHIFT)

#define PLAYER_STEP_HEIGHT_VOXELS 6
#define PLAYER_GROUND_PROBE (VOXEL_SIZE * 0.5f)
#define PLAYER_SUB_STEP (VOXEL_SIZE * 0.25f)
#define PLAYER_MAX_SUB_STEPS 8
#define PLAYER_COYOTE_FRAMES 6
#define PLAYER_MAX_UNSTICK_VOXELS 12
#define PLAYER_SPAWN_SEARCH_METERS 64.0f

// Components below this are treated as no movement at all. Attempting a zero-length move
// "succeeds" without going anywhere, which would silently eat the input.
#define PLAYER_MIN_MOVE 1e-6f
// Voxels are 6.25cm, so an exactly-player-width box catches on stray single voxels along
// an otherwise smooth wall. Pull the collider in a little.
#define PLAYER_COLLIDER_INSET (VOXEL_SIZE * 0.5f)

namespace {
    struct PlayerBox {
        glm::vec3 min, max;
    };

    // PLAYER.pos is the eye/camera position, so the box hangs below it.
    PlayerBox player_box(glm::vec3 pos, float half_width, float height) {
        return {
            glm::vec3(pos.x - half_width, pos.y - half_width, pos.z - height),
            glm::vec3(pos.x + half_width, pos.y + half_width, pos.z),
        };
    }
    PlayerBox player_box(vec3 pos, float half_width, float height) {
        return player_box(glm::vec3(pos.x, pos.y, pos.z), half_width, height);
    }

    bool box_blocked(VoxelWorld *voxel_world, PlayerBox box, glm::vec3 offset = {}, glm::ivec3 *hit = nullptr) {
        return voxel_world_box_is_solid(voxel_world, box.min + offset, box.max + offset, hit);
    }
} // namespace

void player_fix_chunk_offset(Player &PLAYER) {
    PLAYER.prev_unit_offset = PLAYER.player_unit_offset;
    const bool wrap_position = AppSettings::get<settings::Checkbox>("Player", "Wrap Position").value;
    if (wrap_position) {
        PLAYER.player_unit_offset = PLAYER.player_unit_offset + daxa_i32vec3(floor(PLAYER.pos.x), floor(PLAYER.pos.y), floor(PLAYER.pos.z));
        PLAYER.pos = {PLAYER.pos.x - floor(PLAYER.pos.x), PLAYER.pos.y - floor(PLAYER.pos.y), PLAYER.pos.z - floor(PLAYER.pos.z)};
    }
}

void player_startup(Player &PLAYER) {
    PROFILE_FUNC();
    // if (((PLAYER.flags >> 0) & 0x1) != 0) {
    //     return;
    // }

    // toggle fly on. The world doesn't exist yet, so the spawn point gets resolved on
    // the first frame that has one.
    PLAYER.flags = PLAYER_FLAG_FLYING | PLAYER_FLAG_NEEDS_SPAWN;

    AppSettings::add<settings::InputFloat>({"Player", "Movement Speed", {.value = 1.5f}});
    AppSettings::add<settings::InputFloat>({"Player", "Sprint Multiplier", {.value = 3.0f}});
    AppSettings::add<settings::InputFloat>({"Player", "Crouch Multiplier", {.value = 0.5f}});
    AppSettings::add<settings::Checkbox>({"Player", "Wrap Position", {.value = false}});
    AppSettings::add<settings::InputFloat>({"Player", "Jump Strength (meters on Earth)", {.value = 1.0f}});
    AppSettings::add<settings::InputFloat>({"Player", "Height", {.value = 1.75f}});
    AppSettings::add<settings::InputFloat>({"Player", "Crouch Height", {.value = 1.0f}});
    AppSettings::add<settings::InputFloat>({"Player", "Width", {.value = 0.6f}});
    AppSettings::add<settings::InputFloat>({"Player", "Fly Speed Multiplier", {.value = 10.0f}});

    // float ground_level = AppSettings::get<settings::InputFloat>("Atmosphere", "atmosphere_bottom").value * 1000.0f + 2000.0f;
    float ground_level = 0.0f;

    PLAYER.pos = vec3(0.01f, 0.02f, 0.03f + ground_level);
    PLAYER.cam_pos_offset = vec3(0.0);
    PLAYER.vel = vec3(0.0);
    PLAYER.player_unit_offset = daxa_i32vec3(0, 0, 4);
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

void toggle_view(Player &PLAYER) {
    PLAYER.flags = (PLAYER.flags & ~PLAYER_VIEW_MASK) | ((PLAYER.flags & PLAYER_VIEW_MASK) + 1);
    if ((PLAYER.flags & PLAYER_VIEW_MASK) > 1)
        PLAYER.flags = PLAYER.flags & ~PLAYER_VIEW_MASK;
}

void toggle_fly(Player &PLAYER) {
    auto toggled_last_frame = (PLAYER.flags & PLAYER_FLAG_FLY_LATCH) != 0;
    if (!toggled_last_frame) {
        PLAYER.flags ^= PLAYER_FLAG_FLYING;
    }
    PLAYER.flags |= PLAYER_FLAG_FLY_LATCH;
}

vec3 view_vec(Player &PLAYER) {
    switch (PLAYER.flags & PLAYER_VIEW_MASK) {
    case 0: return vec3(0, 0, -0.2f);
    case 1: return (PLAYER.forward * sin(PLAYER.pitch) + vec3(0, 0, cos(-PLAYER.pitch))) * +2.0f;
    default: return vec3(0, 0, 0);
    }
}

#define EARTH_GRAV 9.807f
#define MOON_GRAV 1.625f
#define MARS_GRAV 3.728f
#define JUPITER_GRAV 25.93f

#define GRAVITY EARTH_GRAV

#define EARTH_JUMP_HEIGHT 0.59

void player_perframe(PlayerInput &INPUT, Player &PLAYER) {
    const float mouse_sens = 1.0f;

    // if (INPUT.actions[GAME_ACTION_INTERACT1] != 0) {
    //     PLAYER.roll += INPUT.mouse.pos_delta.x * mouse_sens * INPUT.sensitivity * 0.001f;
    // } else {
    PLAYER.yaw += INPUT.mouse.pos_delta.x * mouse_sens * INPUT.sensitivity * 0.001f;
    PLAYER.pitch -= INPUT.mouse.pos_delta.y * mouse_sens * INPUT.sensitivity * 0.001f;
    // }

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

    PLAYER.forward = move_forward; // std::bit_cast<vec3>(forward);
    PLAYER.lateral = move_lateral; // std::bit_cast<vec3>(lateral);

    if (INPUT.actions[GAME_ACTION_CYCLE_VIEW] != 0) {
        if ((PLAYER.flags & PLAYER_FLAG_VIEW_LATCH) == 0) {
            PLAYER.flags |= PLAYER_FLAG_VIEW_LATCH;
            toggle_view(PLAYER);
        }
    } else {
        PLAYER.flags &= ~PLAYER_FLAG_VIEW_LATCH;
    }

    if (INPUT.actions[GAME_ACTION_TOGGLE_FLY] != 0) {
        toggle_fly(PLAYER);
        PLAYER.vel = vec3(0, 0, 0);
    } else {
        PLAYER.flags &= ~PLAYER_FLAG_FLY_LATCH;
    }

    const bool is_flying = (PLAYER.flags & PLAYER_FLAG_FLYING) != 0;
    const bool is_crouched = (PLAYER.flags & PLAYER_FLAG_CROUCHED) != 0;
    const bool grounded_recently = (PLAYER.flags & PLAYER_COYOTE_MASK) != 0;
    const bool is_third_person = (PLAYER.flags & PLAYER_FLAG_THIRD_PERSON) != 0;

    const float speed = AppSettings::get<settings::InputFloat>("Player", "Movement Speed").value;
    const float sprint_speed = AppSettings::get<settings::InputFloat>("Player", "Sprint Multiplier").value;
    const float crouch_speed = AppSettings::get<settings::InputFloat>("Player", "Crouch Multiplier").value;
    const float jump_strength = AppSettings::get<settings::InputFloat>("Player", "Jump Strength (meters on Earth)").value;
    const float player_height = AppSettings::get<settings::InputFloat>("Player", "Height").value;
    const float crouch_height = AppSettings::get<settings::InputFloat>("Player", "Crouch Height").value;
    const float fly_speed_mult = AppSettings::get<settings::InputFloat>("Player", "Fly Speed Multiplier").value;
    const float half_width = AppSettings::get<settings::InputFloat>("Player", "Width").value * 0.5f;
    const float collide_half_width = std::max(half_width - PLAYER_COLLIDER_INSET, VOXEL_SIZE);
    float height = player_height;

    auto *voxel_world = INPUT.voxel_world;

    if ((PLAYER.flags & PLAYER_FLAG_NEEDS_SPAWN) != 0 && voxel_world != nullptr) {
        PLAYER.flags &= ~PLAYER_FLAG_NEEDS_SPAWN;
        // The startup position isn't guaranteed to be in open air, so walk upwards from
        // it until we find somewhere with headroom to stand and solid ground underfoot.
        const int32_t max_steps = int32_t(PLAYER_SPAWN_SEARCH_METERS * VOXEL_SCL);
        for (int32_t step = 0; step < max_steps; ++step) {
            auto spawn_pos = PLAYER.pos;
            spawn_pos.z += float(step) * VOXEL_SIZE;
            auto box = player_box(spawn_pos, collide_half_width, player_height);
            if (box_blocked(voxel_world, box))
                continue;
            if (!box_blocked(voxel_world, box, glm::vec3(0, 0, -VOXEL_SIZE)))
                continue;
            PLAYER.pos = spawn_pos;
            PLAYER.vel = vec3(0, 0, 0);
            break;
        }
    }

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
    if (is_flying)
        applied_speed *= fly_speed_mult;

    vec3 acc = vec3(0, 0, 0);

    if (is_flying) {
        if (INPUT.actions[GAME_ACTION_JUMP] != 0)
            move_vec = move_vec + vec3(0, 0, 1);
        if (INPUT.actions[GAME_ACTION_CROUCH] != 0)
            move_vec = move_vec - vec3(0, 0, 1);
    } else {
        if (grounded_recently && INPUT.actions[GAME_ACTION_JUMP] != 0) {
            PLAYER.vel.z = EARTH_GRAV * sqrt(jump_strength * 2.0 / EARTH_GRAV);
            // Consume the coyote window, so it can't be spent on a second jump.
            PLAYER.flags &= ~PLAYER_COYOTE_MASK;
        } else {
            acc.z = -GRAVITY;
        }

        if (INPUT.actions[GAME_ACTION_CROUCH] != 0) {
            if (!is_crouched) {
                PLAYER.pos.z -= height - crouch_height;
                PLAYER.cam_pos_offset.z += height - crouch_height;
            }
            height = crouch_height;
            PLAYER.flags |= PLAYER_FLAG_CROUCHED;
        } else {
            // Standing up raises the eye while the feet stay put; refuse if that would
            // put our head in a ceiling.
            bool no_headroom = false;
            if (is_crouched && voxel_world != nullptr) {
                auto stand_pos = PLAYER.pos;
                stand_pos.z += height - crouch_height;
                no_headroom = box_blocked(voxel_world, player_box(stand_pos, collide_half_width, height));
            }
            if (no_headroom) {
                height = crouch_height;
                PLAYER.flags |= PLAYER_FLAG_CROUCHED;
            } else {
                if (is_crouched) {
                    PLAYER.pos.z += height - crouch_height;
                    PLAYER.cam_pos_offset.z -= height - crouch_height;
                }
                PLAYER.flags &= ~PLAYER_FLAG_CROUCHED;
            }
        }
    }

    float dt = glm::min(INPUT.delta_time, 1.0f);

    PLAYER.vel = PLAYER.vel + acc * dt;
    auto vel = PLAYER.vel + move_vec * applied_speed;
    auto offset = vel * dt;

    PLAYER.flags &= ~PLAYER_FLAG_ON_GROUND;

    // Diagnostics for the collision resolution, surfaced in the debug display below.
    const char *collide_state = "flying";
    bool was_embedded = false;
    int32_t sub_steps_taken = 0;
    int32_t sub_steps_wanted = 0;

    if (is_flying || voxel_world == nullptr) {
        PLAYER.pos = PLAYER.pos + offset;
    } else {
        auto p = glm::vec3(PLAYER.pos.x, PLAYER.pos.y, PLAYER.pos.z);
        auto d = glm::vec3(offset.x, offset.y, offset.z);

        auto blocked_at = [&](glm::vec3 at, glm::ivec3 *hit = nullptr) {
            return box_blocked(voxel_world, player_box(at, collide_half_width, height), {}, hit);
        };

        auto try_move = [&](glm::vec3 delta) -> int32_t {
            int32_t max_lift = grounded_recently ? PLAYER_STEP_HEIGHT_VOXELS : 0;
            for (int32_t r = 0; r <= max_lift; ++r) {
                if (!blocked_at(p + delta + glm::vec3(0, 0, float(r) * VOXEL_SIZE)))
                    return r;
            }
            return -1;
        };
        auto apply = [&](glm::vec3 delta, int32_t lift) {
            p += delta;
            if (lift > 0) {
                auto lift_z = float(lift) * VOXEL_SIZE;
                p.z += lift_z;
                // Let the camera lag behind the step so it isn't a visible pop.
                PLAYER.cam_pos_offset.z -= lift_z;
            }
        };

        was_embedded = blocked_at(p);
        if (was_embedded) {
            auto escape = voxel_world_terrain_normal(voxel_world, p - glm::vec3(0, 0, height * 0.5f));
            for (int32_t r = 1; r <= PLAYER_MAX_UNSTICK_VOXELS; ++r) {
                auto push = escape * (float(r) * VOXEL_SIZE);
                auto out = p + push;
                if (blocked_at(out)) {
                    // Straight up as a fallback -- the normal can point into a pocket.
                    push = glm::vec3(0, 0, float(r) * VOXEL_SIZE);
                    out = p + push;
                    if (blocked_at(out))
                        continue;
                }
                PLAYER.cam_pos_offset.z -= push.z;
                p = out;
                PLAYER.vel.z = 0.0f;
                break;
            }
        }

        auto horiz = glm::vec3(d.x, d.y, 0.0f);
        auto horiz_len = glm::length(horiz);
        if (horiz_len > 0.0f) {
            auto sub_count = clamp(int32_t(ceil(horiz_len / PLAYER_SUB_STEP)), 1, PLAYER_MAX_SUB_STEPS);
            auto sub = horiz / float(sub_count);
            sub_steps_wanted = sub_count;
            collide_state = "clear";

            for (int32_t i = 0; i < sub_count; ++i) {
                auto step = sub;
                auto lift = try_move(step);

                if (lift < 0) {
                    auto hit = glm::ivec3(0);
                    blocked_at(p + step, &hit);
                    // Take the hit voxel's lateral position but sample at body height.
                    // The box scan runs bottom-up, so `hit` is usually a voxel down at
                    // foot level whose normal points straight up -- sampling there would
                    // give a normal perpendicular to travel and nothing to slide along.
                    auto probe = (glm::vec3(hit) + 0.5f) * VOXEL_SIZE;
                    probe.z = p.z - height * 0.5f;
                    auto nrm = voxel_world_terrain_normal(voxel_world, probe);
                    auto into = glm::dot(step, nrm);
                    if (into < 0.0f) {
                        auto slid = step - nrm * into;
                        lift = try_move(slid);
                        if (lift >= 0) {
                            step = slid;
                            collide_state = "slid";
                        }
                    }
                }

                if (lift < 0) {
                    // Last resort: keep whichever single axis still works, so we scrape
                    // along walls the normal couldn't resolve us off of. Skip a component
                    // that's essentially zero -- try_move on a zero delta just re-tests
                    // where we already are and reports success without moving, which
                    // would eat the input for as long as the key is held.
                    if (std::abs(sub.x) > PLAYER_MIN_MOVE) {
                        auto only_x = glm::vec3(sub.x, 0.0f, 0.0f);
                        lift = try_move(only_x);
                        if (lift >= 0) {
                            step = only_x;
                            collide_state = "axis x";
                        }
                    }
                    if (lift < 0 && std::abs(sub.y) > PLAYER_MIN_MOVE) {
                        auto only_y = glm::vec3(0.0f, sub.y, 0.0f);
                        lift = try_move(only_y);
                        if (lift >= 0) {
                            step = only_y;
                            collide_state = "axis y";
                        }
                    }
                }

                if (lift < 0) {
                    collide_state = "blocked";
                    break;
                }
                apply(step, lift);
                ++sub_steps_taken;
            }
        }

        bool grounded = false;

        // Step-down. Walking off a small lip while grounded should follow the ground
        // rather than launch us into a fall -- otherwise descending slopes flicker in and
        // out of grounded, which disables step-up and reads as catching on nothing.
        if (grounded_recently && d.z <= 0.0f) {
            const float max_drop = float(PLAYER_STEP_HEIGHT_VOXELS) * VOXEL_SIZE;
            auto probe = p;
            float dropped = 0.0f;
            while (dropped < max_drop) {
                auto next = probe - glm::vec3(0, 0, PLAYER_SUB_STEP);
                if (blocked_at(next)) {
                    grounded = true;
                    break;
                }
                probe = next;
                dropped += PLAYER_SUB_STEP;
            }
            if (grounded) {
                PLAYER.cam_pos_offset.z += p.z - probe.z;
                p = probe;
                PLAYER.vel.z = 0.0f;
            }
        }

        // Vertical in sub-voxel increments, so we come to rest flush on the surface
        // rather than hovering by however far we happened to fall this frame.
        if (!grounded && d.z != 0.0f) {
            float dir = d.z > 0.0f ? 1.0f : -1.0f;
            float remaining = std::abs(d.z);
            while (remaining > 0.0f) {
                auto step = std::min(remaining, PLAYER_SUB_STEP);
                auto next = p + glm::vec3(0, 0, dir * step);
                if (blocked_at(next)) {
                    PLAYER.vel.z = 0.0f;
                    break;
                }
                p = next;
                remaining -= step;
            }
        }

        // Probe just below the feet rather than keying off a blocked descent, so we
        // stay grounded while standing still too.
        if (!grounded && blocked_at(p - glm::vec3(0, 0, PLAYER_GROUND_PROBE)))
            grounded = true;

        PLAYER.pos = vec3(p.x, p.y, p.z);
        if (grounded)
            PLAYER.flags |= PLAYER_FLAG_ON_GROUND;
    }

    {
        // Coyote timer: refresh while grounded, otherwise tick down.
        auto coyote = (PLAYER.flags & PLAYER_COYOTE_MASK) >> PLAYER_COYOTE_SHIFT;
        if ((PLAYER.flags & PLAYER_FLAG_ON_GROUND) != 0) {
            coyote = PLAYER_COYOTE_FRAMES;
        } else if (coyote > 0) {
            --coyote;
        }
        PLAYER.flags = (PLAYER.flags & ~PLAYER_COYOTE_MASK) | (coyote << PLAYER_COYOTE_SHIFT);
    }

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
    PLAYER.cam.view_to_clip.y.y = -1.0f / tan_half_fov;
    PLAYER.cam.view_to_clip.z.z = +0.0f;
    PLAYER.cam.view_to_clip.z.w = -1.0f;
    PLAYER.cam.view_to_clip.w.z = near;

    PLAYER.cam.clip_to_view = mat4{};
    PLAYER.cam.clip_to_view.x.x = tan_half_fov * aspect;
    PLAYER.cam.clip_to_view.y.y = -tan_half_fov;
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
    const float interp_speed = std::max(length(PLAYER.cam_pos_offset) * float(VOXEL_SCL), 0.25f);
    PLAYER.cam_pos_offset = PLAYER.cam_pos_offset - cam_pos_offset_sign * dt * interp_speed;

    vec3 new_cam_pos_offset_sign = sign(PLAYER.cam_pos_offset);
    if (new_cam_pos_offset_sign.x != cam_pos_offset_sign.x)
        PLAYER.cam_pos_offset.x = 0.0f;
    if (new_cam_pos_offset_sign.y != cam_pos_offset_sign.y)
        PLAYER.cam_pos_offset.y = 0.0f;
    if (new_cam_pos_offset_sign.z != cam_pos_offset_sign.z)
        PLAYER.cam_pos_offset.z = 0.0f;

    auto cam_pos = PLAYER.pos + PLAYER.cam_pos_offset + view_vec(PLAYER);

    PLAYER.cam.view_to_sample = std::bit_cast<mat4>(clip_to_sample * std::bit_cast<glm::mat4>(PLAYER.cam.view_to_clip));
    PLAYER.cam.sample_to_view = std::bit_cast<mat4>(std::bit_cast<glm::mat4>(PLAYER.cam.clip_to_view) * sample_to_clip);

    PLAYER.cam.view_to_world = std::bit_cast<mat4>(translation_matrix(cam_pos) * rotation_matrix(PLAYER.yaw + float(M_PI) * is_third_person, PLAYER.pitch, PLAYER.roll));
    PLAYER.cam.world_to_view = std::bit_cast<mat4>(inv_rotation_matrix(PLAYER.yaw + float(M_PI) * is_third_person, PLAYER.pitch, PLAYER.roll) * translation_matrix(cam_pos * -1.0f));

    PLAYER.cam.clip_to_prev_clip = std::bit_cast<mat4>(
        std::bit_cast<glm::mat4>(PLAYER.cam.prev_view_to_prev_clip) *
        std::bit_cast<glm::mat4>(PLAYER.cam.prev_world_to_prev_view) *
        std::bit_cast<glm::mat4>(PLAYER.cam.view_to_world) *
        std::bit_cast<glm::mat4>(PLAYER.cam.clip_to_view));

    // debug_utils::DebugDisplay::set_debug_string("Player Pos", format("%.3f, %.3f, %.3f", double(PLAYER.pos.x), double(PLAYER.pos.y), double(PLAYER.pos.z)).data);
    // debug_utils::DebugDisplay::set_debug_string("Player Pos (camera)", format("%.3f, %.3f, %.3f", double(cam_pos.x), double(cam_pos.y), double(cam_pos.z)).data);
    // debug_utils::DebugDisplay::set_debug_string("Player Pos (voxel)", format("%.3f, %.3f, %.3f", double(PLAYER.pos.x * VOXEL_SCL), double(PLAYER.pos.y * VOXEL_SCL), double(PLAYER.pos.z * VOXEL_SCL)).data);
    // debug_utils::DebugDisplay::set_debug_string("Player Rot (Y/P/R)", format("%.3f, %.3f, %.3f", double(PLAYER.yaw), double(PLAYER.pitch), double(PLAYER.roll)).data);
    // debug_utils::DebugDisplay::set_debug_string("Player Unit Offset", format("%d, %d, %d", PLAYER.player_unit_offset.x, PLAYER.player_unit_offset.y, PLAYER.player_unit_offset.z).data);
    // debug_utils::DebugDisplay::set_debug_string("Player Vel (m/s)", format("%.3f, %.3f, %.3f", double(vel.x), double(vel.y), double(vel.z)).data);
    // debug_utils::DebugDisplay::set_debug_string("Player On Ground", ((PLAYER.flags & PLAYER_FLAG_ON_GROUND) != 0) ? "true" : "false");
    // debug_utils::DebugDisplay::set_debug_string("Player Collide", format("%s %d/%d%s", collide_state, sub_steps_taken, sub_steps_wanted, was_embedded ? " EMBEDDED" : "").data);
}

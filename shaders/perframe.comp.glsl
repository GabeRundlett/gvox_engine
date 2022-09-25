#include <shared/shared.inl>

DAXA_USE_PUSH_CONSTANT(PerframeCompPush)

#include <utils/voxel.glsl>
#include <utils/raytrace.glsl>

#define PLAYER_HEIGHT 1.8
#define COLLIDE_DELTA 0.09

#if 1 // REALISTIC
#define PLAYER_SPEED 1.5
#define PLAYER_ACCEL 10.0
#define EARTH_JUMP_HEIGHT 0.59
#else // MINECRAFT
#define PLAYER_SPEED 2.5
#define PLAYER_ACCEL 10.0
#define EARTH_JUMP_HEIGHT 1.1
#endif

#define EARTH_GRAV 9.807
#define MOON_GRAV 1.625
#define MARS_GRAV 3.728
#define JUPITER_GRAV 25.93

#define GRAVITY EARTH_GRAV

void toggle_view() {
    PLAYER.view_state = (PLAYER.view_state & ~0xf) | ((PLAYER.view_state & 0xf) + 1);
    if ((PLAYER.view_state & 0xf) > 1)
        PLAYER.view_state = (PLAYER.view_state & ~0xf) | 0;
}
void toggle_fly() {
    u32 is_flying = (PLAYER.view_state >> 6) & 0x1;
    PLAYER.view_state = (PLAYER.view_state & ~(0x1 << 6)) | ((1 - is_flying) << 6);
}

f32vec3 view_vec() {
    switch (PLAYER.view_state & 0xf) {
    case 0: return PLAYER.forward * 0.32;
    case 1: return PLAYER.cam.rot_mat * f32vec3(0, -2, 0);
    default: return f32vec3(0, 0, 0);
    }
}

b32 get_flag(u32 index) {
    return ((INPUT.settings.flags >> index) & 0x01) == 0x01;
}

void perframe_player() {
    const f32 mouse_sens = 1.0;

    b32 is_flying = ((PLAYER.view_state >> 6) & 0x1) == 1;

    if (is_flying) {
        PLAYER.speed = PLAYER_SPEED * 30;
        PLAYER.accel_rate = PLAYER_ACCEL * 30;
        PLAYER.on_ground = true;
    } else {
        if (PLAYER.pos.z <= 0) {
            PLAYER.pos.z = 0;
            PLAYER.on_ground = true;
            PLAYER.force_vel.z = 0;
            PLAYER.move_vel.z = 0;
        }
        PLAYER.speed = PLAYER_SPEED;
        PLAYER.accel_rate = PLAYER_ACCEL;
        if (INPUT.keyboard.keys[GAME_KEY_JUMP] != 0 && PLAYER.on_ground)
            PLAYER.force_vel.z = EARTH_GRAV * sqrt(EARTH_JUMP_HEIGHT * 2.0 / EARTH_GRAV);
    }
    PLAYER.sprint_speed = 2.5;

    if (INPUT.keyboard.keys[GAME_KEY_CYCLE_VIEW] != 0) {
        if ((PLAYER.view_state & 0x10) == 0) {
            PLAYER.view_state |= 0x10;
            toggle_view();
        }
    } else {
        PLAYER.view_state &= ~0x10;
    }
    if (INPUT.keyboard.keys[GAME_KEY_TOGGLE_FLY] != 0) {
        if ((PLAYER.view_state & 0x20) == 0) {
            PLAYER.view_state |= 0x20;
            toggle_fly();
        }
    } else {
        PLAYER.view_state &= ~0x20;
    }

    PLAYER.rot.z += INPUT.mouse.pos_delta.x * mouse_sens * 0.001;
    PLAYER.rot.x -= INPUT.mouse.pos_delta.y * mouse_sens * 0.001;

    const float MAX_ROT = 1.57;
    if (PLAYER.rot.x > MAX_ROT)
        PLAYER.rot.x = MAX_ROT;
    if (PLAYER.rot.x < -MAX_ROT)
        PLAYER.rot.x = -MAX_ROT;
    float sin_rot_x = sin(PLAYER.rot.x), cos_rot_x = cos(PLAYER.rot.x);
    float sin_rot_y = sin(PLAYER.rot.y), cos_rot_y = cos(PLAYER.rot.y);
    float sin_rot_z = sin(PLAYER.rot.z), cos_rot_z = cos(PLAYER.rot.z);

    // clang-format off
    PLAYER.cam.rot_mat = 
        f32mat3x3(
            cos_rot_z, -sin_rot_z, 0,
            sin_rot_z,  cos_rot_z, 0,
            0,          0,         1
        ) *
        f32mat3x3(
            1,          0,          0,
            0,  cos_rot_x,  sin_rot_x,
            0, -sin_rot_x,  cos_rot_x
        );
    // clang-format on

    f32vec3 move_vec = f32vec3(0, 0, 0);
    PLAYER.forward = f32vec3(+sin_rot_z, +cos_rot_z, 0);
    PLAYER.lateral = f32vec3(+cos_rot_z, -sin_rot_z, 0);

    if (INPUT.keyboard.keys[GAME_KEY_MOVE_FORWARD] != 0)
        move_vec += PLAYER.forward;
    if (INPUT.keyboard.keys[GAME_KEY_MOVE_BACKWARD] != 0)
        move_vec -= PLAYER.forward;
    if (INPUT.keyboard.keys[GAME_KEY_MOVE_LEFT] != 0)
        move_vec -= PLAYER.lateral;
    if (INPUT.keyboard.keys[GAME_KEY_MOVE_RIGHT] != 0)
        move_vec += PLAYER.lateral;

    if (is_flying) {
        if (INPUT.keyboard.keys[GAME_KEY_JUMP] != 0)
            move_vec += f32vec3(0, 0, 1);
        if (INPUT.keyboard.keys[GAME_KEY_CROUCH] != 0)
            move_vec -= f32vec3(0, 0, 1);
    }

    f32 applied_accel = PLAYER.accel_rate;
    if (INPUT.keyboard.keys[GAME_KEY_SPRINT] != 0)
        PLAYER.max_speed += INPUT.delta_time * PLAYER.accel_rate;
    else
        PLAYER.max_speed -= INPUT.delta_time * PLAYER.accel_rate;
    PLAYER.max_speed = clamp(PLAYER.max_speed, PLAYER.speed, PLAYER.speed * PLAYER.sprint_speed);

    f32 move_magsq = dot(move_vec, move_vec);
    if (move_magsq > 0) {
        move_vec = normalize(move_vec) * INPUT.delta_time * applied_accel * (PLAYER.on_ground ? 1 : 0.1);
    }

    f32 move_vel_mag = length(PLAYER.move_vel);
    f32vec3 move_vel_dir;
    if (move_vel_mag > 0) {
        move_vel_dir = PLAYER.move_vel / move_vel_mag;
        if (PLAYER.on_ground)
            PLAYER.move_vel -= move_vel_dir * min(PLAYER.accel_rate * 0.4 * INPUT.delta_time, move_vel_mag);
    }
    PLAYER.move_vel += move_vec * 2;

    move_vel_mag = length(PLAYER.move_vel);
    if (move_vel_mag > 0) {
        PLAYER.move_vel = PLAYER.move_vel / move_vel_mag * min(move_vel_mag, PLAYER.max_speed);
    }

    if (is_flying) {
        PLAYER.force_vel = f32vec3(0, 0, 0);
    } else {
        PLAYER.force_vel += f32vec3(0, 0, -1) * GRAVITY * INPUT.delta_time;
    }

    f32vec3 vel = PLAYER.move_vel + PLAYER.force_vel;
    PLAYER.pos += vel * INPUT.delta_time;
    PLAYER.on_ground = true;

    f32vec3 cam_offset = view_vec();

    if (INPUT.mouse.scroll_delta.y < 0.0) {
        PLAYER.edit_radius *= 1.05;
    } else if (INPUT.mouse.scroll_delta.y > 0.0) {
        PLAYER.edit_radius /= 1.05;
    }

    PLAYER.edit_radius = clamp(PLAYER.edit_radius, 1.0 / VOXEL_SCL, 127.0 / VOXEL_SCL);

    PLAYER.cam.pos = PLAYER.pos + f32vec3(0, 0, PLAYER_HEIGHT - 0.3) + cam_offset;
    PLAYER.cam.fov = INPUT.settings.fov * 3.14159 / 180.0;
    PLAYER.cam.tan_half_fov = tan(PLAYER.cam.fov * 0.5);

    Ray ray;
    ray.o = PLAYER.pos + f32vec3(0, 0, 0);
    f32vec3 player_ray_o;

#if 1
    // TODO: rewrite collisions to be swept AABB
    player_ray_o = PLAYER.pos + f32vec3(0.0, 0.0, 0.1);
    {
        f32 axis_sign = sign(vel.z);
        if (axis_sign != 0) {
            ray.nrm = f32vec3(0, 0, axis_sign);
            ray.inv_nrm = 1.0 / ray.nrm;
            IntersectionRecord intersections[4];
            ray.o = player_ray_o + f32vec3(-COLLIDE_DELTA, -COLLIDE_DELTA, 0);
            intersections[0] = intersect_chunk(ray);
            ray.o = player_ray_o + f32vec3(+COLLIDE_DELTA, -COLLIDE_DELTA, 0);
            intersections[1] = intersect_chunk(ray);
            ray.o = player_ray_o + f32vec3(-COLLIDE_DELTA, +COLLIDE_DELTA, 0);
            intersections[2] = intersect_chunk(ray);
            ray.o = player_ray_o + f32vec3(+COLLIDE_DELTA, +COLLIDE_DELTA, 0);
            intersections[3] = intersect_chunk(ray);
            IntersectionRecord intersection;
            intersection.hit = intersections[0].hit || intersections[1].hit || intersections[2].hit || intersections[3].hit;
            intersection.dist = min(min(intersections[0].dist, intersections[1].dist), min(intersections[2].dist, intersections[3].dist));
            PLAYER.on_ground = false;
            if (intersection.hit && intersection.dist <= 0.1) {
                PLAYER.on_ground = true;
                PLAYER.pos.z = ray.o.z + ray.nrm.z * intersection.dist - ray.nrm.z * 0.1 - 0.1;
                PLAYER.move_vel.z = 0, PLAYER.force_vel.z = 0;
            }
        }
    }
    player_ray_o = PLAYER.pos + f32vec3(0.0, 0.0, 0.1);
    {
        f32 axis_sign = sign(vel.x);
        if (axis_sign != 0) {
            ray.nrm = f32vec3(axis_sign, 0, 0);
            ray.inv_nrm = 1.0 / ray.nrm;
            IntersectionRecord intersections[4];
            ray.o = player_ray_o + f32vec3(0, -COLLIDE_DELTA, -COLLIDE_DELTA);
            intersections[0] = intersect_chunk(ray);
            ray.o = player_ray_o + f32vec3(0, +COLLIDE_DELTA, -COLLIDE_DELTA);
            intersections[1] = intersect_chunk(ray);
            ray.o = player_ray_o + f32vec3(0, -COLLIDE_DELTA, +COLLIDE_DELTA);
            intersections[2] = intersect_chunk(ray);
            ray.o = player_ray_o + f32vec3(0, +COLLIDE_DELTA, +COLLIDE_DELTA);
            intersections[3] = intersect_chunk(ray);
            IntersectionRecord intersection;
            intersection.hit = intersections[0].hit || intersections[1].hit || intersections[2].hit || intersections[3].hit;
            intersection.dist = min(min(intersections[0].dist, intersections[1].dist), min(intersections[2].dist, intersections[3].dist));
            if (intersection.hit) {
                if (intersection.dist <= 0.1) {
                    f32vec3 hit_pos = ray.o + ray.nrm * (intersection.dist + 0.001);
                    u32 temp_chunk_index;
                    if (sample_lod(hit_pos + f32vec3(0, 0, 1) / VOXEL_SCL, temp_chunk_index) != 0) {
                        PLAYER.pos.z = floor(PLAYER.pos.z * VOXEL_SCL + 1.001) / VOXEL_SCL;
                    } else {
                        PLAYER.pos.x = ray.o.x + ray.nrm.x * intersection.dist - ray.nrm.x * 0.1;
                        PLAYER.move_vel.x = 0, PLAYER.force_vel.x = 0;
                    }
                }
            }
        }
    }
    player_ray_o = PLAYER.pos + f32vec3(0.0, 0.0, 0.1);
    {
        f32 axis_sign = sign(vel.y);
        if (axis_sign != 0) {
            ray.nrm = f32vec3(0, axis_sign, 0);
            ray.inv_nrm = 1.0 / ray.nrm;
            IntersectionRecord intersections[4];
            ray.o = player_ray_o + f32vec3(-COLLIDE_DELTA, 0, -COLLIDE_DELTA);
            intersections[0] = intersect_chunk(ray);
            ray.o = player_ray_o + f32vec3(+COLLIDE_DELTA, 0, -COLLIDE_DELTA);
            intersections[1] = intersect_chunk(ray);
            ray.o = player_ray_o + f32vec3(-COLLIDE_DELTA, 0, +COLLIDE_DELTA);
            intersections[2] = intersect_chunk(ray);
            ray.o = player_ray_o + f32vec3(+COLLIDE_DELTA, 0, +COLLIDE_DELTA);
            intersections[3] = intersect_chunk(ray);
            IntersectionRecord intersection;
            intersection.hit = intersections[0].hit || intersections[1].hit || intersections[2].hit || intersections[3].hit;
            intersection.dist = min(min(intersections[0].dist, intersections[1].dist), min(intersections[2].dist, intersections[3].dist));
            if (intersection.hit) {
                if (intersection.dist <= 0.1) {
                    f32vec3 hit_pos = ray.o + ray.nrm * (intersection.dist + 0.001);
                    u32 temp_chunk_index;
                    if (sample_lod(hit_pos + f32vec3(0, 0, 1) / VOXEL_SCL, temp_chunk_index) != 0) {
                        PLAYER.pos.z = floor(PLAYER.pos.z * VOXEL_SCL + 1.001) / VOXEL_SCL;
                    } else {
                        PLAYER.pos.y = ray.o.y + ray.nrm.y * intersection.dist - ray.nrm.y * 0.1;
                        PLAYER.move_vel.y = 0, PLAYER.force_vel.y = 0;
                    }
                }
            }
        }
    }
#endif
}

u32 calculate_chunk_edit() {
    u32 non_chunkgen_update_n = 0;
    for (i32 zi = 0; zi < 5; ++zi) {
        if (VOXEL_WORLD.chunk_update_n >= 128)
            break;
        for (i32 yi = 0; yi < 5; ++yi) {
            if (VOXEL_WORLD.chunk_update_n >= 128)
                break;
            for (i32 xi = 0; xi < 5; ++xi) {
                if (VOXEL_WORLD.chunk_update_n >= 128)
                    break;
                u32 i = get_chunk_index(get_chunk_i(get_voxel_i(GLOBALS.pick_pos + (i32vec3(xi, yi, zi) - 2) * CHUNK_SIZE / VOXEL_SCL)));
                if (VOXEL_WORLD.chunks_genstate[i].edit_stage == 2) { // TODO: determine if a chunk actually needs to be modified.
                    VOXEL_WORLD.chunks_genstate[i].edit_stage = 3;
                    VOXEL_WORLD.chunk_update_indices[VOXEL_WORLD.chunk_update_n] = i;
                    ++VOXEL_WORLD.chunk_update_n;
                    ++non_chunkgen_update_n;
                }
            }
        }
    }
    if (VOXEL_WORLD.chunk_update_n > 128)
        VOXEL_WORLD.chunk_update_n = 128;
    return non_chunkgen_update_n;
}

void perframe_voxel_world() {
    for (u32 i = 0; i < VOXEL_WORLD.chunk_update_n; ++i) {
        u32 chunk_index = VOXEL_WORLD.chunk_update_indices[i];
        if (VOXEL_WORLD.chunks_genstate[chunk_index].edit_stage == 1)
            VOXEL_WORLD.chunks_genstate[chunk_index].edit_stage = 2;
        if (VOXEL_WORLD.chunks_genstate[chunk_index].edit_stage == 3)
            VOXEL_WORLD.chunks_genstate[chunk_index].edit_stage = 2;
    }
    VOXEL_WORLD.chunk_update_n = 0;

    VOXEL_WORLD.center_pt = PLAYER.pos;
    f32 min_dist_sq = 1000.0;

    for (u32 zi = 0; zi < CHUNK_NZ; ++zi) {
        for (u32 yi = 0; yi < CHUNK_NY; ++yi) {
            for (u32 xi = 0; xi < CHUNK_NX; ++xi) {
                i32vec3 chunk_i = i32vec3(xi, yi, zi);
                u32 i = get_chunk_index(chunk_i);
#if USING_BRICKMAP
                f32vec3 box_center = f32vec3(0);
#else
                f32vec3 box_center = (VOXEL_CHUNKS[i].box.bound_max + VOXEL_CHUNKS[i].box.bound_min) * 0.5;
#endif
                f32vec3 del = VOXEL_WORLD.center_pt - box_center;
                f32 dist_sq = dot(del, del);
                u32 stage = VOXEL_WORLD.chunks_genstate[i].edit_stage;
                if (stage == 0 && (dist_sq < min_dist_sq || VOXEL_WORLD.chunk_update_n == 0)) {
                    min_dist_sq = dist_sq;
                    VOXEL_WORLD.chunkgen_index = i;
                    VOXEL_WORLD.chunk_update_n = 1;
                    VOXEL_WORLD.chunk_update_indices[0] = i;
                }
            }
        }
    }

    u32 non_chunkgen_update_n = 0;
    if (!get_flag(GPU_INPUT_FLAG_INDEX_LIMIT_EDIT_RATE) || INPUT.time - PLAYER.last_edit_time > INPUT.settings.edit_rate) {
        if (INPUT.mouse.buttons[GAME_MOUSE_BUTTON_LEFT] != 0) {
            GLOBALS.edit_flags = 1;
            non_chunkgen_update_n = calculate_chunk_edit();
            PLAYER.edit_voxel_id = BlockID_Air;
            PLAYER.last_edit_time = INPUT.time;
        } else if (INPUT.mouse.buttons[GAME_MOUSE_BUTTON_RIGHT] != 0) {
            GLOBALS.edit_flags = 2;
            non_chunkgen_update_n = calculate_chunk_edit();
            PLAYER.edit_voxel_id = BlockID_Stone;
            PLAYER.last_edit_time = INPUT.time;
        } else {
            GLOBALS.edit_flags = 0;
        }
    } else {
        GLOBALS.edit_flags = 0;
    }
    if (INPUT.mouse.buttons[GAME_MOUSE_BUTTON_LEFT] == 0 &&
        INPUT.mouse.buttons[GAME_MOUSE_BUTTON_RIGHT] == 0)
        PLAYER.last_edit_time = 0.0;

    for (u32 i = 0; i < VOXEL_WORLD.chunk_update_n; ++i) {
        u32 chunk_index = VOXEL_WORLD.chunk_update_indices[i];
        if (VOXEL_WORLD.chunks_genstate[chunk_index].edit_stage == 0)
            VOXEL_WORLD.chunks_genstate[chunk_index].edit_stage = 1;
    }

    INDIRECT.chunk_edit_dispatch = u32vec3((CHUNK_SIZE + 7) / 8);
    INDIRECT.subchunk_x2x4_dispatch = u32vec3(1, 64, 1);
    INDIRECT.subchunk_x8up_dispatch = u32vec3(1, 1, 1);

    INDIRECT.chunk_edit_dispatch.z *= non_chunkgen_update_n;
    INDIRECT.subchunk_x2x4_dispatch.y *= VOXEL_WORLD.chunk_update_n;
    INDIRECT.subchunk_x8up_dispatch.y *= VOXEL_WORLD.chunk_update_n;
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    u32 prev_edit_flags = GLOBALS.edit_flags;

    perframe_player();
    perframe_voxel_world();

    f32vec2 pick_uv = f32vec2(0.0, 0.0);
    if (get_flag(GPU_INPUT_FLAG_INDEX_PAUSED)) {
        f32vec2 pixel_p = INPUT.mouse.pos;
        f32vec2 frame_dim = INPUT.frame_dim;
        f32vec2 inv_frame_dim = f32vec2(1.0, 1.0) / frame_dim;
        f32 aspect = frame_dim.x * inv_frame_dim.y;
        pick_uv = pixel_p * inv_frame_dim;
        pick_uv = (pick_uv - 0.5) * f32vec2(aspect, 1.0) * 2.0;
    }

    Ray pick_ray = create_view_ray(pick_uv);
    IntersectionRecord pick_intersection = intersect_chunk(pick_ray);

    GLOBALS.pick_pos = pick_ray.o + pick_ray.nrm * pick_intersection.dist;
    GLOBALS.pick_nrm = pick_intersection.nrm;

    if (GLOBALS.edit_flags == 0) {
        GLOBALS.edit_origin = round(GLOBALS.pick_pos * VOXEL_SCL) / VOXEL_SCL;
    } else if (GLOBALS.edit_flags != 0 && prev_edit_flags == 0) {
        GLOBALS.edit_origin = round(GLOBALS.pick_pos * VOXEL_SCL) / VOXEL_SCL;
    }

    SCENE.capsules[0].p0 = PLAYER.pos + f32vec3(0, 0, 0.3);
    SCENE.capsules[0].p1 = PLAYER.pos + f32vec3(0, 0, PLAYER_HEIGHT - 0.3);
}

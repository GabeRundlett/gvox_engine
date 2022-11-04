#include <shared/shared.inl>

DAXA_USE_PUSH_CONSTANT(PerframeCompPush)

#include <utils/voxel.glsl>
#include <utils/raytrace.glsl>

#include <brush_info.glsl>

#define PLAYER_HEIGHT 1.8
#define PLAYER_HEAD_RADIUS (0.6 / 2)
#define COLLIDE_DELTA 0.09

#if 1 // REALISTIC
#define PLAYER_SPEED 1.5
#define PLAYER_SPRINT_MUL 2.5
#define PLAYER_ACCEL 10.0
#define EARTH_JUMP_HEIGHT 0.59
#else // MINECRAFT-LIKE
#define PLAYER_SPEED 4.3
#define PLAYER_SPRINT_MUL 1.3
#define PLAYER_ACCEL 30.0
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
void toggle_brush() {
    u32 brush_enabled = (PLAYER.view_state >> 8) & 0x1;
    PLAYER.view_state = (PLAYER.view_state & ~(0x1 << 8)) | ((1 - brush_enabled) << 8);
}

f32vec3 view_vec() {
    switch (PLAYER.view_state & 0xf) {
    case 0: return PLAYER.forward * (PLAYER_HEAD_RADIUS + 0.001);
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

    if (PLAYER.pos.z <= 0) {
        PLAYER.pos.z = 0;
        PLAYER.on_ground = true;
        PLAYER.force_vel.z = 0;
        PLAYER.move_vel.z = 0;
    }

    if (is_flying) {
        PLAYER.speed = PLAYER_SPEED * 30;
        PLAYER.accel_rate = PLAYER_ACCEL * 30;
        PLAYER.on_ground = true;
    } else {
        PLAYER.speed = PLAYER_SPEED;
        PLAYER.accel_rate = PLAYER_ACCEL;
        if (INPUT.keyboard.keys[GAME_KEY_JUMP] != 0 && PLAYER.on_ground)
            PLAYER.force_vel.z = EARTH_GRAV * sqrt(EARTH_JUMP_HEIGHT * 2.0 / EARTH_GRAV);
    }
    PLAYER.sprint_speed = 2.5;

    PLAYER.rot.z += INPUT.mouse.pos_delta.x * mouse_sens * INPUT.settings.sensitivity * 0.001;
    PLAYER.rot.x -= INPUT.mouse.pos_delta.y * mouse_sens * INPUT.settings.sensitivity * 0.001;

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

    f32 applied_accel = PLAYER.accel_rate;

    if (INPUT.settings.tool_id == GAME_TOOL_BRUSH) {
        if (INPUT.keyboard.keys[GAME_KEY_TOGGLE_BRUSH] != 0) {
            if ((PLAYER.view_state & (1 << 7)) == 0) {
                PLAYER.view_state |= (1 << 7);
                toggle_brush();
            }
        } else {
            PLAYER.view_state &= ~(1 << 7);
        }
    }

    if (!get_flag(GPU_INPUT_FLAG_INDEX_PAUSED)) {
        if (INPUT.keyboard.keys[GAME_KEY_CYCLE_VIEW] != 0) {
            if ((PLAYER.view_state & (1 << 4)) == 0) {
                PLAYER.view_state |= (1 << 4);
                toggle_view();
            }
        } else {
            PLAYER.view_state &= ~(1 << 4);
        }
        if (INPUT.keyboard.keys[GAME_KEY_TOGGLE_FLY] != 0) {
            if ((PLAYER.view_state & (1 << 5)) == 0) {
                PLAYER.view_state |= (1 << 5);
                toggle_fly();
            }
        } else {
            PLAYER.view_state &= ~(1 << 5);
        }

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

        if (INPUT.keyboard.keys[GAME_KEY_SPRINT] != 0)
            PLAYER.max_speed += INPUT.delta_time * PLAYER.accel_rate;
        else
            PLAYER.max_speed -= INPUT.delta_time * PLAYER.accel_rate;
    }

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

    PLAYER.edit_radius = clamp(PLAYER.edit_radius, 1.0 / VOXEL_SCL, 64.0 / VOXEL_SCL);

    PLAYER.cam.pos = PLAYER.pos + f32vec3(0, 0, PLAYER_HEIGHT - PLAYER_HEAD_RADIUS);

    f32 cam_offset_len = length(cam_offset);

    Ray cam_offset_ray;
    cam_offset_ray.o = PLAYER.cam.pos;
    cam_offset_ray.nrm = cam_offset / cam_offset_len;
    cam_offset_ray.inv_nrm = 1.0 / cam_offset_ray.nrm;

    IntersectionRecord offset_intersection = intersect_voxels(cam_offset_ray);
    if (offset_intersection.hit)
        cam_offset_len = min(cam_offset_len, offset_intersection.dist - 0.001);
    cam_offset = cam_offset_ray.nrm * cam_offset_len;

    PLAYER.cam.pos += cam_offset;
    PLAYER.cam.fov = INPUT.settings.fov * 3.14159 / 180.0;
    PLAYER.cam.tan_half_fov = tan(PLAYER.cam.fov * 0.5);

    Ray ray;
    ray.o = PLAYER.pos + f32vec3(0, 0, 0);
    f32vec3 player_ray_o;

    SCENE.capsules[0].r = PLAYER_HEAD_RADIUS;
    SCENE.capsules[0].p0 = PLAYER.pos + f32vec3(0, 0, PLAYER_HEAD_RADIUS);
    SCENE.capsules[0].p1 = PLAYER.pos + f32vec3(0, 0, PLAYER_HEIGHT - PLAYER_HEAD_RADIUS);

    // f32vec3 head_p = PLAYER.pos + f32vec3(0, 0, 1.7);
    // f32vec3 shoulder_p = PLAYER.pos + f32vec3(0, 0, 1.5);
    // f32vec3 chest_p = PLAYER.pos + f32vec3(0, 0, 1.4) + PLAYER.forward * 0.01;
    // f32vec3 waist_p = PLAYER.pos + f32vec3(0, 0, 1.02);
    // f32vec3 knee_p = PLAYER.pos + f32vec3(0, 0, 0.54);
    // f32vec3 hand_p = PLAYER.pos + f32vec3(0, 0, 0.8);

    // SCENE.capsules[0].r = PLAYER_HEAD_RADIUS;
    // SCENE.capsules[0].p0 = head_p - f32vec3(0, 0, PLAYER_HEAD_RADIUS * 0.2);
    // SCENE.capsules[0].p1 = head_p + f32vec3(0, 0, PLAYER_HEAD_RADIUS * 0.2);
    // SCENE.capsules[1].r = PLAYER_HEAD_RADIUS * 0.6;
    // SCENE.capsules[1].p0 = head_p;
    // SCENE.capsules[1].p1 = waist_p;
    // SCENE.capsules[2].r = PLAYER_HEAD_RADIUS * 0.7;
    // SCENE.capsules[2].p0 = shoulder_p - PLAYER.lateral * 0.2;
    // SCENE.capsules[2].p1 = shoulder_p + PLAYER.lateral * 0.2;
    // SCENE.capsules[3].r = PLAYER_HEAD_RADIUS * 0.8;
    // SCENE.capsules[3].p0 = chest_p - PLAYER.lateral * 0.1;
    // SCENE.capsules[3].p1 = chest_p + PLAYER.lateral * 0.1;
    // SCENE.capsules[4].r = PLAYER_HEAD_RADIUS * 0.8;
    // SCENE.capsules[4].p0 = chest_p - PLAYER.lateral * 0.1;
    // SCENE.capsules[4].p1 = waist_p - PLAYER.lateral * 0.04;
    // SCENE.capsules[5].r = PLAYER_HEAD_RADIUS * 0.8;
    // SCENE.capsules[5].p0 = chest_p + PLAYER.lateral * 0.1;
    // SCENE.capsules[5].p1 = waist_p + PLAYER.lateral * 0.04;
    // SCENE.capsules[6].r = PLAYER_HEAD_RADIUS * 0.8;
    // SCENE.capsules[6].p0 = waist_p - PLAYER.lateral * 0.08;
    // SCENE.capsules[6].p1 = waist_p + PLAYER.lateral * 0.08;
    // SCENE.capsules[7].r = 0.085;
    // SCENE.capsules[7].p0 = waist_p - PLAYER.lateral * 0.08 - f32vec3(0, 0, 0.06);
    // SCENE.capsules[7].p1 = knee_p - PLAYER.lateral * 0.10 + f32vec3(0, 0, 0.05);
    // SCENE.capsules[8].r = 0.085;
    // SCENE.capsules[8].p0 = waist_p + PLAYER.lateral * 0.08 - f32vec3(0, 0, 0.06);
    // SCENE.capsules[8].p1 = knee_p + PLAYER.lateral * 0.10 + f32vec3(0, 0, 0.05);
    // SCENE.capsules[9].r = 0.07;
    // SCENE.capsules[9].p0 = knee_p - PLAYER.lateral * 0.10 + f32vec3(0, 0, 0.05);
    // SCENE.capsules[9].p1 = PLAYER.pos - PLAYER.lateral * 0.08 + f32vec3(0, 0, 0.07);
    // SCENE.capsules[10].r = 0.07;
    // SCENE.capsules[10].p0 = knee_p + PLAYER.lateral * 0.10 + f32vec3(0, 0, 0.05);
    // SCENE.capsules[10].p1 = PLAYER.pos + PLAYER.lateral * 0.08 + f32vec3(0, 0, 0.07);
    // SCENE.capsules[11].r = 0.05;
    // SCENE.capsules[11].p0 = shoulder_p - PLAYER.lateral * 0.22;
    // SCENE.capsules[11].p1 = shoulder_p - PLAYER.lateral * 0.24 - f32vec3(0, 0, 0.25);
    // SCENE.capsules[12].r = 0.05;
    // SCENE.capsules[12].p0 = shoulder_p + PLAYER.lateral * 0.22;
    // SCENE.capsules[12].p1 = shoulder_p + PLAYER.lateral * 0.24 - f32vec3(0, 0, 0.25);
    // SCENE.capsules[13].r = 0.045;
    // SCENE.capsules[13].p0 = shoulder_p - PLAYER.lateral * 0.24 - f32vec3(0, 0, 0.25);
    // SCENE.capsules[13].p1 = hand_p - PLAYER.lateral * 0.22;
    // SCENE.capsules[14].r = 0.045;
    // SCENE.capsules[14].p0 = shoulder_p + PLAYER.lateral * 0.24 - f32vec3(0, 0, 0.25);
    // SCENE.capsules[14].p1 = hand_p + PLAYER.lateral * 0.22;

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
            intersections[0] = intersect_voxels(ray);
            ray.o = player_ray_o + f32vec3(+COLLIDE_DELTA, -COLLIDE_DELTA, 0);
            intersections[1] = intersect_voxels(ray);
            ray.o = player_ray_o + f32vec3(-COLLIDE_DELTA, +COLLIDE_DELTA, 0);
            intersections[2] = intersect_voxels(ray);
            ray.o = player_ray_o + f32vec3(+COLLIDE_DELTA, +COLLIDE_DELTA, 0);
            intersections[3] = intersect_voxels(ray);
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
            intersections[0] = intersect_voxels(ray);
            ray.o = player_ray_o + f32vec3(0, +COLLIDE_DELTA, -COLLIDE_DELTA);
            intersections[1] = intersect_voxels(ray);
            ray.o = player_ray_o + f32vec3(0, -COLLIDE_DELTA, +COLLIDE_DELTA);
            intersections[2] = intersect_voxels(ray);
            ray.o = player_ray_o + f32vec3(0, +COLLIDE_DELTA, +COLLIDE_DELTA);
            intersections[3] = intersect_voxels(ray);
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
            intersections[0] = intersect_voxels(ray);
            ray.o = player_ray_o + f32vec3(+COLLIDE_DELTA, 0, -COLLIDE_DELTA);
            intersections[1] = intersect_voxels(ray);
            ray.o = player_ray_o + f32vec3(-COLLIDE_DELTA, 0, +COLLIDE_DELTA);
            intersections[2] = intersect_voxels(ray);
            ray.o = player_ray_o + f32vec3(+COLLIDE_DELTA, 0, +COLLIDE_DELTA);
            intersections[3] = intersect_voxels(ray);
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
    for (i32 zi = 0; zi < WORLD_CHUNK_NZ; ++zi) {
        if (VOXEL_WORLD.chunk_update_n >= MAX_CHUNK_UPDATES)
            break;
        for (i32 yi = 0; yi < WORLD_CHUNK_NY; ++yi) {
            if (VOXEL_WORLD.chunk_update_n >= MAX_CHUNK_UPDATES)
                break;
            for (i32 xi = 0; xi < WORLD_CHUNK_NX; ++xi) {
                if (VOXEL_WORLD.chunk_update_n >= MAX_CHUNK_UPDATES)
                    break;
                u32 i = get_chunk_index_WORLD(i32vec3(xi, yi, zi));
                Box chunk_box = VOXEL_WORLD.voxel_chunks[i].box;
                if (VOXEL_WORLD.chunks_genstate[i].edit_stage == 2 && overlaps(chunk_box, SCENE.pick_box)) {
                    VOXEL_WORLD.chunks_genstate[i].edit_stage = 3;
                    VOXEL_WORLD.chunk_update_indices[VOXEL_WORLD.chunk_update_n] = i;
                    ++VOXEL_WORLD.chunk_update_n;
                    ++non_chunkgen_update_n;
                }
            }
        }
    }
    if (VOXEL_WORLD.chunk_update_n > MAX_CHUNK_UPDATES)
        VOXEL_WORLD.chunk_update_n = MAX_CHUNK_UPDATES;
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

    for (u32 zi = 0; zi < WORLD_CHUNK_NZ; ++zi) {
        for (u32 yi = 0; yi < WORLD_CHUNK_NY; ++yi) {
            for (u32 xi = 0; xi < WORLD_CHUNK_NX; ++xi) {
                i32vec3 chunk_i = i32vec3(xi, yi, zi);
                u32 i = get_chunk_index_WORLD(chunk_i);
                f32vec3 box_center = (VOXEL_WORLD.voxel_chunks[i].box.bound_max + VOXEL_WORLD.voxel_chunks[i].box.bound_min) * 0.5;
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

    VOXEL_WORLD.brush_chunkgen_index = (VOXEL_WORLD.brush_chunkgen_index - WORLD_CHUNK_N + 1) % BRUSH_CHUNK_N + WORLD_CHUNK_N;

    // if (VOXEL_WORLD.chunk_update_n == 0) {
    //     for (u32 i = WORLD_CHUNK_N; i < WORLD_CHUNK_N + BRUSH_CHUNK_N; ++i) {
    //         u32 stage = VOXEL_WORLD.chunks_genstate[i].edit_stage;
    //         if (stage == 0) {
    //             VOXEL_WORLD.chunkgen_index = i;
    //             VOXEL_WORLD.chunk_update_n = 1;
    //             VOXEL_WORLD.chunk_update_indices[0] = i;
    //             break;
    //         }
    //     }
    // }

    u32 non_chunkgen_update_n = 0;
    b32 brush_enabled = ((PLAYER.view_state >> 8) & 0x1) == 1;

    if (GLOBALS.pick_intersection.hit && brush_enabled) {
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
    }

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

    if (INPUT.mouse.buttons[GAME_MOUSE_BUTTON_LEFT] == 0 &&
        INPUT.mouse.buttons[GAME_MOUSE_BUTTON_RIGHT] == 0)
        PLAYER.last_edit_time = 0.0;
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    u32 prev_edit_flags = GLOBALS.edit_flags;

    perframe_player();
    perframe_voxel_world();

    b32 brush_enabled = INPUT.settings.tool_id == GAME_TOOL_BRUSH && ((PLAYER.view_state >> 8) & 0x1) == 1;

    if (brush_enabled) {
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
        GLOBALS.pick_intersection = intersect_voxels(pick_ray);

        GLOBALS.brush_offset = custom_brush_origin_offset();

        if (GLOBALS.pick_intersection.hit) {
            f32vec3 p0 = pick_ray.o + pick_ray.nrm * GLOBALS.pick_intersection.dist;
            f32vec3 p1 = p0 - GLOBALS.pick_intersection.nrm * 0.01 / VOXEL_SCL;
            p0 = p0 + GLOBALS.pick_intersection.nrm * 0.01 / VOXEL_SCL;
            GLOBALS.brush_origin = p0 + GLOBALS.brush_offset;
            GLOBALS.pick_intersection.hit = custom_brush_enable(p0, p1);
        }

        Box custom_box = custom_brush_box();

        // SCENE.pick_box.bound_min = custom_box.bound_min + GLOBALS.brush_origin - GLOBALS.brush_offset;
        // SCENE.pick_box.bound_max = custom_box.bound_max + GLOBALS.brush_origin - GLOBALS.brush_offset;

        // SCENE.pick_box.bound_min = floor((custom_box.bound_min + GLOBALS.brush_origin - GLOBALS.brush_offset) * VOXEL_SCL) / VOXEL_SCL;
        // SCENE.pick_box.bound_max = floor((custom_box.bound_max + GLOBALS.brush_origin - GLOBALS.brush_offset) * VOXEL_SCL) / VOXEL_SCL;

        SCENE.pick_box.bound_min = round((custom_box.bound_min + GLOBALS.brush_origin - GLOBALS.brush_offset) * VOXEL_SCL) / VOXEL_SCL;
        SCENE.pick_box.bound_max = round((custom_box.bound_max + GLOBALS.brush_origin - GLOBALS.brush_offset) * VOXEL_SCL) / VOXEL_SCL;

        if (prev_edit_flags == 0 && INPUT.keyboard.keys[GAME_KEY_INTERACT0] != 0) {
            GLOBALS.edit_origin = round(GLOBALS.brush_origin * VOXEL_SCL) / VOXEL_SCL;
        }

        SCENE.brush_origin_sphere.o = GLOBALS.edit_origin;
        SCENE.brush_origin_sphere.r = 0.25;
    }
}

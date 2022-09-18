#include <shared/shared.inl>

DAXA_USE_PUSH_CONSTANT(PerframeCompPush)

#include <utils/voxel.glsl>
#include <utils/raytrace.glsl>

void toggle_view() {
    PLAYER.view_state = (PLAYER.view_state & ~0xf) | ((PLAYER.view_state & 0xf) + 1);
    if ((PLAYER.view_state & 0xf) > 1)
        PLAYER.view_state = (PLAYER.view_state & ~0xf) | 0;
}

f32vec3 view_vec() {
    switch (PLAYER.view_state) {
    case 0: return PLAYER.forward * 0.55;
    case 1: return PLAYER.cam.rot_mat * f32vec3(0, -2, 0);
    default: return f32vec3(0, 0, 0);
    }
}

void perframe_player() {
    const f32 mouse_sens = 1.0;
    const f32 speed_mul = 30.0;
    const f32 sprint_mul = 5.0;

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

    if (INPUT.keyboard.keys[GAME_KEY_F5] != 0) {
        if ((PLAYER.view_state & 0x10) == 0) {
            PLAYER.view_state |= 0x10;
            toggle_view();
        }
    } else {
        PLAYER.view_state &= ~0x10;
    }

    f32vec3 move_vec = f32vec3(0, 0, 0);
    PLAYER.forward = f32vec3(+sin_rot_z, +cos_rot_z, 0);
    PLAYER.lateral = f32vec3(+cos_rot_z, -sin_rot_z, 0);
    f32 speed = speed_mul;

    if (INPUT.keyboard.keys[GAME_KEY_W] != 0)
        move_vec += PLAYER.forward;
    if (INPUT.keyboard.keys[GAME_KEY_S] != 0)
        move_vec -= PLAYER.forward;
    if (INPUT.keyboard.keys[GAME_KEY_A] != 0)
        move_vec -= PLAYER.lateral;
    if (INPUT.keyboard.keys[GAME_KEY_D] != 0)
        move_vec += PLAYER.lateral;

    if (INPUT.keyboard.keys[GAME_KEY_SPACE] != 0)
        move_vec += f32vec3(0, 0, 1);
    if (INPUT.keyboard.keys[GAME_KEY_LEFT_CONTROL] != 0)
        move_vec -= f32vec3(0, 0, 1);

    if (INPUT.keyboard.keys[GAME_KEY_LEFT_SHIFT] != 0)
        speed *= sprint_mul;

    PLAYER.pos += move_vec * INPUT.delta_time * speed;

    f32vec3 cam_offset = view_vec();

    if (INPUT.mouse.scroll_delta.y < 0.0) {
        PLAYER.edit_radius *= 1.05;
    } else if (INPUT.mouse.scroll_delta.y > 0.0) {
        PLAYER.edit_radius /= 1.05;
    }

    PLAYER.edit_radius = clamp(PLAYER.edit_radius, 1.5 / VOXEL_SCL, 64 / VOXEL_SCL);

    PLAYER.cam.pos = PLAYER.pos + f32vec3(0, 0, 2) + cam_offset;
    PLAYER.cam.tan_half_fov = tan(INPUT.fov * 3.14159 / 360.0);
}

u32 calculate_chunk_edit() {
    u32 non_chunkgen_update_n = 0;
    i32vec3 chunk_i = get_chunk_i(get_voxel_i(GLOBALS.pick_pos));
    for (i32 zi = 0; zi < 3; ++zi) {
        for (i32 yi = 0; yi < 3; ++yi) {
            for (i32 xi = 0; xi < 3; ++xi) {
                u32 i = get_chunk_index(chunk_i + i32vec3(xi, yi, zi) - 1);
                if (VOXEL_WORLD.chunks_genstate[i].edit_stage == 2) {
                    VOXEL_WORLD.chunks_genstate[i].edit_stage = 3;
                    VOXEL_WORLD.chunk_update_indices[VOXEL_WORLD.chunk_update_n] = i;
                    ++VOXEL_WORLD.chunk_update_n;
                    ++non_chunkgen_update_n;
                }
            }
        }
    }
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
                f32vec3 box_center = (VOXEL_CHUNKS[i].box.bound_max + VOXEL_CHUNKS[i].box.bound_min) * 0.5;
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
    if (INPUT.time - PLAYER.last_edit_time > 0.2) {
        if (INPUT.mouse.buttons[GAME_MOUSE_BUTTON_LEFT] != 0) {
            non_chunkgen_update_n = calculate_chunk_edit();
            PLAYER.edit_voxel_id = BlockID_Air;
            PLAYER.last_edit_time = INPUT.time;
        } else if (INPUT.mouse.buttons[GAME_MOUSE_BUTTON_RIGHT] != 0) {
            non_chunkgen_update_n = calculate_chunk_edit();
            PLAYER.edit_voxel_id = BlockID_Stone;
            PLAYER.last_edit_time = INPUT.time;
        }

        if (INPUT.mouse.buttons[GAME_MOUSE_BUTTON_LEFT] == 0 &&
            INPUT.mouse.buttons[GAME_MOUSE_BUTTON_RIGHT] == 0)
            PLAYER.last_edit_time = 0.0;
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
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    perframe_player();
    perframe_voxel_world();

    Ray pick_ray = create_view_ray(f32vec2(0.0, 0.0));
    IntersectionRecord pick_intersection = intersect_chunk(pick_ray);

    GLOBALS.pick_pos = pick_ray.o + pick_ray.nrm * pick_intersection.dist;
    GLOBALS.pick_nrm = pick_intersection.nrm;

    SCENE.capsules[0].p0 = PLAYER.pos + f32vec3(0, 0, 0.3);
    SCENE.capsules[0].p1 = PLAYER.pos + f32vec3(0, 0, 1.7);
}

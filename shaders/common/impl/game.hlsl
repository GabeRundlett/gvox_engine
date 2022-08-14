#pragma once

#include "common/interface/game.hlsl"
#include "common/impl/player.hlsl"
#include "common/impl/voxel_world.hlsl"

#define EDIT_RADIUS 1

void ShapeScene::trace(in out GameTraceState trace_state, Ray ray, int shape_type) {
    for (uint shape_i = 0; shape_i < shape_n; ++shape_i) {
        Shape shape = shapes[shape_i];
        TraceRecord shape_trace_result;
        switch (shape.type) {
        case ShapeType::Sphere: shape_trace_result = trace_sphere(ray, spheres[shape.storage_index]); break;
        case ShapeType::Box: shape_trace_result = trace_box(ray, boxes[shape.storage_index]); break;
        case ShapeType::Capsule: shape_trace_result = trace_capsule(ray, capsules[shape.storage_index]); break;
        case ShapeType::SdfWorld: shape_trace_result = trace_sdf_world(ray, sdf_worlds[shape.storage_index]); break;
        default: break;
        }
        if (shape_trace_result.hit && shape_trace_result.dist < trace_state.trace_record.dist) {
            trace_state.trace_record = shape_trace_result;
            trace_state.shape_i = shape_i;
            trace_state.shape_type = shape_type;
        }
    }
}

void ShapeScene::eval_color(in out GameTraceState trace_state) {
    switch (shapes[trace_state.shape_i].material_id) {
    case 0: {
        Capsule c = capsules[shapes[trace_state.shape_i].storage_index];
        float3 local_pos = trace_state.draw_sample.pos - c.p1;
        float2x2 rot_mat = float2x2(float2(-c.forward.y, c.forward.x), c.forward);
        local_pos.xy = mul(rot_mat, local_pos.xy);

#define MINION 0
        float2 uv = local_pos.xz;
        float2 e1_uv = uv - float2(0.1, 0.0);
        float e1 = dot(e1_uv, e1_uv) > 0.001;
        float2 e2_uv = uv + float2(0.1, 0.0);
        float e2 = dot(e2_uv, e2_uv) > 0.001;

#if MINION
        float2 m_uv = uv + float2(0.0, 0.2);
#else
        float2 m_uv = uv + float2(0.0, 0.04);
#endif
        float m = clamp((dot(m_uv, m_uv) > 0.02) + (m_uv.y > -0.05), 0, 1);

        float face_fac = clamp(e1 * e2 * m + (local_pos.y < 0) * 10, 0, 1);
        float pants_fac = local_pos.z > -0.6;

        float3 result;
#if MINION
        result = float3(0.90, 0.78, 0.075);
#else
        result = float3(0.60, 0.27, 0.20);
#endif

        float radial = atan2(local_pos.y, local_pos.x) / 6.28 + 0.5;
        float2 b_pocket_pos = float2(abs(abs(radial - 0.5) - 0.25) - 0.075, (local_pos.z + 0.7) * 0.5);
        float2 f_pocket_pos = float2(abs(abs(radial - 0.5) - 0.25) - 0.200, (local_pos.z + 0.7) * 0.5);
        float b_pockets_fac = b_pocket_pos.y < 0.0 && b_pocket_pos.x < 0.04 && b_pocket_pos.x > -0.04 && dot(b_pocket_pos, b_pocket_pos) < 0.003 && local_pos.y < 0;
        float f_pockets_fac = f_pocket_pos.y < 0.0 && f_pocket_pos.x < 0.02 && f_pocket_pos.x > -0.06 && dot(f_pocket_pos, f_pocket_pos) < 0.003 && local_pos.y > 0;
        float belt_fac = (fmod(radial * 20, 1) < 0.8 && local_pos.z > -0.64 && local_pos.z < -0.62);

#if !MINION
        float2 shirt_uv = float2(local_pos.x, (local_pos.z + 0.40) * 4);
        float shirt_fac = shirt_uv.y > 0 || (dot(shirt_uv, shirt_uv) < 0.1 + local_pos.y * 0.1);
        result = lerp(float3(0.4, 0.05, 0.042), result, shirt_fac);
#else
        float goggles_fac = ((dot(e1_uv, e1_uv) > 0.013) && (dot(e2_uv, e2_uv) > 0.013)) || (local_pos.y < 0);
        result = lerp(float3(0.3, 0.3, 0.3), result, goggles_fac);
        float whites_eyes_fac = ((dot(e1_uv, e1_uv) > 0.008) && (dot(e2_uv, e2_uv) > 0.008)) || (local_pos.y < 0);
        result = lerp(float3(1.0, 1.0, 1.0), result, whites_eyes_fac);
#endif
        result = lerp(float3(0.04, 0.04, 0.12), result, pants_fac);
        result = lerp(result, float3(0.03, 0.03, 0.10), b_pockets_fac || f_pockets_fac || (f_pocket_pos.x > 0.045 && local_pos.z < -0.68 && local_pos.z > -1.5));
        result = lerp(float3(0.0, 0.0, 0.0), result, face_fac);
        result = lerp(result, float3(0.04, 0.02, 0.01), belt_fac);

        trace_state.draw_sample.col = result;
    } break;
    case 1: {
        float s = 50;
        float mx = smoothstep(0.45, 0.55, s * 0.5 - abs(s * 0.5 - frac(trace_state.draw_sample.pos.x) * s));
        float my = smoothstep(0.45, 0.55, s * 0.5 - abs(s * 0.5 - frac(trace_state.draw_sample.pos.y) * s));
        float mz = smoothstep(0.45, 0.55, s * 0.5 - abs(s * 0.5 - frac(trace_state.draw_sample.pos.z) * s));
        float3 m = lerp(float3(1, 0, 0), float3(0.01, 0.01, 0.01), mx) / 3 +
                   lerp(float3(0, 1, 0), float3(0.01, 0.01, 0.01), my) / 3 +
                   lerp(float3(0, 0, 1), float3(0.01, 0.01, 0.01), mz) / 3;
        trace_state.draw_sample.col = m;
    } break;
    case 2:
        trace_state.draw_sample.col = float3(1, 1, 1);
        break;
    }
}

void ShapeScene::add_shape(Sphere s, float3 color, uint material_id) {
    if (sphere_n < MAX_DEBUG_SPHERES) {
        spheres[sphere_n] = s;

        shapes[shape_n].type = ShapeType::Sphere;
        shapes[shape_n].material_id = material_id;
        shapes[shape_n].storage_index = sphere_n;
        shapes[shape_n].color = color;

        ++sphere_n;
        ++shape_n;
    }
}

void ShapeScene::add_shape(Box b, float3 color, uint material_id) {
    if (box_n < MAX_DEBUG_BOXES) {
        boxes[box_n] = b;

        shapes[shape_n].type = ShapeType::Box;
        shapes[shape_n].storage_index = box_n;
        shapes[shape_n].material_id = material_id;
        shapes[shape_n].color = color;

        ++box_n;
        ++shape_n;
    }
}

void ShapeScene::add_shape(Capsule c, float3 color, uint material_id) {
    if (capsule_n < MAX_DEBUG_CAPSULES) {
        capsules[capsule_n] = c;

        shapes[shape_n].type = ShapeType::Capsule;
        shapes[shape_n].storage_index = capsule_n;
        shapes[shape_n].material_id = material_id;
        shapes[shape_n].color = color;

        ++capsule_n;
        ++shape_n;
    }
}

void ShapeScene::add_shape(SdfWorld s, float3 color, uint material_id) {
    if (capsule_n < MAX_DEBUG_CAPSULES) {
        sdf_worlds[sdf_world_n] = s;
        shapes[shape_n].type = ShapeType::SdfWorld;
        shapes[shape_n].storage_index = sdf_world_n;
        shapes[shape_n].material_id = material_id;
        shapes[shape_n].color = color;
        ++sdf_world_n;
        ++shape_n;
    }
}

static const float3 sun_col = float3(1, 0.85, 0.5) * 4;

float3 sample_sky(float3 nrm) {
    float sky_val = clamp(dot(nrm, float3(0, 0, -1)) * 0.5 + 0.5, 0, 1);
    return lerp(float3(0.02, 0.05, 0.90) * 2, float3(0.08, 0.10, 0.54), pow(sky_val, 2));
}

void Game::init() {
    Box temp_box;
    Sphere temp_sphere;
    Capsule temp_capsule;
    SdfWorld temp_sdf_world;

    // temp_box.bound_min = float3(0, 0, 0) - float3(1, 1, 2.0);
    // temp_box.bound_max = float3(0, 0, 0) + float3(1, 1, 2.0);
    // collidable_scene.add_shape(temp_box, float3(0.8, 0.8, 0.8), 1);
    // temp_box.bound_min = float3(0, 0, 0) - float3(10, 1, 0);
    // temp_box.bound_max = float3(0, 0, 0) + float3(-5, 0, 0.5);
    // collidable_scene.add_shape(temp_box, float3(0.8, 0.8, 0.8), 1);

    // temp_box.bound_min = float3(0, 0, 0);
    // temp_box.bound_max = float3(0, 0, 0);
    // debug_scene.add_shape(temp_box, float3(0.2, 1.0, 1.0), 0);

    // temp_box.bound_min = float3(0, 0, 0);
    // temp_box.bound_max = float3(0, 0, 0);
    // debug_scene.add_shape(temp_box, float3(0.2, 1.0, 1.0), 0);

    temp_capsule.p0 = float3(0, 0, 0);
    temp_capsule.p1 = float3(0, 0, 0);
    temp_capsule.r = 0.02;
    temp_capsule.forward = float2(0, 1);
    debug_scene.add_shape(temp_capsule, float3(0.9, 0.7, 0.4), 0);

    debug_scene.add_shape(temp_sdf_world, float3(0, 0, 0), 2);

    // temp_sphere.o = float3(0, 0, 0);
    // temp_sphere.r = 0.02;
    // debug_scene.add_shape(temp_sphere, float3(0.9, 0.7, 0.4), 0);

    // temp_sphere.o = float3(0, 0, 0);
    // temp_sphere.r = 0.02;
    // debug_scene.add_shape(temp_sphere, float3(0.9, 0.7, 0.4), 0);
    // collision rays
    // debug_scene.add_shape(temp_capsule, float3(0.1, 0.1, 0.1), 0);
    // debug_scene.add_shape(temp_capsule, float3(0.9, 0.1, 0.1), 0);
    // debug_scene.add_shape(temp_capsule, float3(0.1, 0.9, 0.1), 0);
    // debug_scene.add_shape(temp_capsule, float3(0.9, 0.9, 0.1), 0);
    // debug_scene.add_shape(temp_capsule, float3(0.1, 0.1, 0.9), 0);
    // debug_scene.add_shape(temp_capsule, float3(0.9, 0.1, 0.9), 0);
    // debug_scene.add_shape(temp_capsule, float3(0.1, 0.9, 0.9), 0);
    // debug_scene.add_shape(temp_capsule, float3(0.9, 0.9, 0.9), 0);

    player.init();

    voxel_world.center_pt = player.pos;
    voxel_world.init();

    // temp_box = voxel_world.box;
    // temp_box.bound_min += float3(1, 1, 1) * 0.25;
    // temp_box.bound_max -= float3(1, 1, 1) * 0.25;
    // debug_scene.add_shape(temp_box, float3(0, 1, 0.5), 0);

    sun_nrm = normalize(float3(-1.5, -3, 2));
    // editing = false;
}

TraceRecord Game::cube_min_trace(Ray ray, Box box) {
    TraceRecord final_trace_record;
    GameTraceState trace_state;
    float3 ray_origins[] = {
        float3(box.bound_min.x, box.bound_min.y, box.bound_min.z),
        float3(box.bound_max.x, box.bound_min.y, box.bound_min.z),
        float3(box.bound_min.x, box.bound_max.y, box.bound_min.z),
        float3(box.bound_max.x, box.bound_max.y, box.bound_min.z),
        float3(box.bound_min.x, box.bound_min.y, box.bound_max.z),
        float3(box.bound_max.x, box.bound_min.y, box.bound_max.z),
        float3(box.bound_min.x, box.bound_max.y, box.bound_max.z),
        float3(box.bound_max.x, box.bound_max.y, box.bound_max.z),
    };

    for (int i = 0; i < 8; ++i) {
        ray.o = ray_origins[i];
        trace_state.default_init();
        trace_collidables(trace_state, ray);
        TraceRecord trace_record = trace_state.trace_record;
        // debug_scene.capsules[i + 1].p0 = ray.o;
        // debug_scene.capsules[i + 1].p1 = ray.o + ray.nrm * trace_record.dist * trace_record.hit;
        if (trace_record.hit && trace_record.dist < final_trace_record.dist) {
            final_trace_record = trace_record;
        }
    }
    return final_trace_record;
}

void Game::update(in out Input input) {
    if (input.keyboard.keys[GLFW_KEY_R] != 0) {
        default_init();
        init();
    }

    player.calc_update(input);

    voxel_world.center_pt = player.pos;
    voxel_world.update(input);

    float3 player_v = (player.move_vel + player.force_vel) * input.delta_time;
    float3 p = player.pos;
    Ray ray;
    ray.o = p + float3(0, 0, 0);
    float3 vel = player.move_vel + player.force_vel;

    player.apply_update(player_v, input);

    Ray pick_ray;
    pick_ray.o = player.camera.pos;
    pick_ray.nrm = mul(player.camera.rot_mat, float3(0, 1, 0));
    pick_ray.inv_nrm = 1.0 / pick_ray.nrm;
    float pick_depth = trace_depth(pick_ray, input.max_steps());

    if (pick_depth > EDIT_RADIUS * 1.5) {
        pick_pos[0] = player.camera.pos + pick_ray.nrm * (pick_depth + 1.0 / VOXEL_SCL);
        pick_pos[1] = player.camera.pos + pick_ray.nrm * (pick_depth - 1.0 / VOXEL_SCL);

        if (input.mouse.buttons[GLFW_MOUSE_BUTTON_RIGHT] != 0) {
            voxel_world.edit_info.pos = pick_pos[1];
            voxel_world.edit_info.block_id = BlockID::Stone;
        } else if (input.mouse.buttons[GLFW_MOUSE_BUTTON_LEFT] != 0) {
            voxel_world.edit_info.pos = pick_pos[0];
            voxel_world.edit_info.block_id = BlockID::Air;
        }
        voxel_world.edit_info.radius = EDIT_RADIUS;
        voxel_world.edit_info.col = input.block_color;
        voxel_world.queue_edit();
    } else {
        pick_pos[0] = -1000;
        pick_pos[1] = -1000;
    }

    // debug_scene.boxes[0].bound_min = player.pos - float3(0.1, 0.1, 0.0);
    // debug_scene.boxes[0].bound_max = player.pos + float3(0.1, 0.1, 0.2);
    debug_scene.capsules[0].p0 = player.pos + float3(0, 0, 0.3);
    debug_scene.capsules[0].p1 = player.pos + float3(0, 0, PLAYER_EYE_HEIGHT);
    debug_scene.capsules[0].r = 0.3;
    if (dot(vel.xy, vel.xy) > 0.001) {
        debug_scene.capsules[0].forward = normalize(vel.xy);
    }
    // debug_scene.spheres[1].o = player.pos;

#define COLLIDE_DELTA 0.09
    float3 player_ray_o;

    player_ray_o = player.pos + float3(0.0, 0.0, 0.1);
    {
        float axis_sign = sign(vel.z);
        if (axis_sign != 0) {
            ray.nrm = float3(0, 0, axis_sign);
            ray.inv_nrm = 1.0 / ray.nrm;
            GameTraceState trace_state;
            TraceRecord trace_records[4];
            ray.o = player_ray_o + float3(-COLLIDE_DELTA, -COLLIDE_DELTA, 0);
            trace_state.default_init();
            trace_collidables(trace_state, ray);
            trace_records[0] = trace_state.trace_record;
            ray.o = player_ray_o + float3(+COLLIDE_DELTA, -COLLIDE_DELTA, 0);
            trace_state.default_init();
            trace_collidables(trace_state, ray);
            trace_records[1] = trace_state.trace_record;
            ray.o = player_ray_o + float3(-COLLIDE_DELTA, +COLLIDE_DELTA, 0);
            trace_state.default_init();
            trace_collidables(trace_state, ray);
            trace_records[2] = trace_state.trace_record;
            ray.o = player_ray_o + float3(+COLLIDE_DELTA, +COLLIDE_DELTA, 0);
            trace_state.default_init();
            trace_collidables(trace_state, ray);
            trace_records[3] = trace_state.trace_record;
            TraceRecord trace_record;
            trace_record.hit = trace_records[0].hit || trace_records[1].hit || trace_records[2].hit || trace_records[3].hit;
            trace_record.dist = min(min(trace_records[0].dist, trace_records[1].dist), min(trace_records[2].dist, trace_records[3].dist));
            player.on_ground = false;
            if (trace_record.hit && trace_record.dist <= 0.1) {
                player.on_ground = true;
                player.pos.z = ray.o.z + ray.nrm.z * trace_record.dist - ray.nrm.z * 0.1 - 0.1;
                player.move_vel.z = 0, player.force_vel.z = 0;
            }
        }
    }
    player_ray_o = player.pos + float3(0.0, 0.0, 0.1);
    {
        float axis_sign = sign(vel.x);
        if (axis_sign != 0) {
            ray.nrm = float3(axis_sign, 0, 0);
            ray.inv_nrm = 1.0 / ray.nrm;
            GameTraceState trace_state;
            TraceRecord trace_records[4];
            ray.o = player_ray_o + float3(0, -COLLIDE_DELTA, -COLLIDE_DELTA);
            trace_state.default_init();
            trace_collidables(trace_state, ray);
            trace_records[0] = trace_state.trace_record;
            ray.o = player_ray_o + float3(0, +COLLIDE_DELTA, -COLLIDE_DELTA);
            trace_state.default_init();
            trace_collidables(trace_state, ray);
            trace_records[1] = trace_state.trace_record;
            ray.o = player_ray_o + float3(0, -COLLIDE_DELTA, +COLLIDE_DELTA);
            trace_state.default_init();
            trace_collidables(trace_state, ray);
            trace_records[2] = trace_state.trace_record;
            ray.o = player_ray_o + float3(0, +COLLIDE_DELTA, +COLLIDE_DELTA);
            trace_state.default_init();
            trace_collidables(trace_state, ray);
            trace_records[3] = trace_state.trace_record;
            TraceRecord trace_record;
            trace_record.hit = trace_records[0].hit || trace_records[1].hit || trace_records[2].hit || trace_records[3].hit;
            trace_record.dist = min(min(trace_records[0].dist, trace_records[1].dist), min(trace_records[2].dist, trace_records[3].dist));
            if (trace_record.hit) {
                if (trace_record.dist <= 0.1) {
                    float3 hit_pos = ray.o + ray.nrm * (trace_record.dist + 0.001);
                    // debug_scene.boxes[1].bound_min = hit_pos - float3(0.05, 0.05, 0.05);
                    // debug_scene.boxes[1].bound_max = hit_pos + float3(0.05, 0.05, 0.05);
                    uint temp_chunk_index;
                    if (voxel_world.sample_lod(hit_pos + float3(0, 0, 1) / VOXEL_SCL, temp_chunk_index) != 0) {
                        player.pos.z = floor(player.pos.z * VOXEL_SCL + 1.001) / VOXEL_SCL;
                    } else {
                        player.pos.x = ray.o.x + ray.nrm.x * trace_record.dist - ray.nrm.x * 0.1;
                        player.move_vel.x = 0, player.force_vel.x = 0;
                    }
                }
            }
        }
    }
    player_ray_o = player.pos + float3(0.0, 0.0, 0.1);
    {
        float axis_sign = sign(vel.y);
        if (axis_sign != 0) {
            ray.nrm = float3(0, axis_sign, 0);
            ray.inv_nrm = 1.0 / ray.nrm;
            GameTraceState trace_state;
            TraceRecord trace_records[4];
            ray.o = player_ray_o + float3(-COLLIDE_DELTA, 0, -COLLIDE_DELTA);
            trace_state.default_init();
            trace_collidables(trace_state, ray);
            trace_records[0] = trace_state.trace_record;
            ray.o = player_ray_o + float3(+COLLIDE_DELTA, 0, -COLLIDE_DELTA);
            trace_state.default_init();
            trace_collidables(trace_state, ray);
            trace_records[1] = trace_state.trace_record;
            ray.o = player_ray_o + float3(-COLLIDE_DELTA, 0, +COLLIDE_DELTA);
            trace_state.default_init();
            trace_collidables(trace_state, ray);
            trace_records[2] = trace_state.trace_record;
            ray.o = player_ray_o + float3(+COLLIDE_DELTA, 0, +COLLIDE_DELTA);
            trace_state.default_init();
            trace_collidables(trace_state, ray);
            trace_records[3] = trace_state.trace_record;
            TraceRecord trace_record;
            trace_record.hit = trace_records[0].hit || trace_records[1].hit || trace_records[2].hit || trace_records[3].hit;
            trace_record.dist = min(min(trace_records[0].dist, trace_records[1].dist), min(trace_records[2].dist, trace_records[3].dist));
            if (trace_record.hit) {
                if (trace_record.dist <= 0.1) {
                    float3 hit_pos = ray.o + ray.nrm * (trace_record.dist + 0.001);
                    uint temp_chunk_index;
                    if (voxel_world.sample_lod(hit_pos + float3(0, 0, 1) / VOXEL_SCL, temp_chunk_index) != 0) {
                        player.pos.z = floor(player.pos.z * VOXEL_SCL + 1.001) / VOXEL_SCL;
                    } else {
                        player.pos.y = ray.o.y + ray.nrm.y * trace_record.dist - ray.nrm.y * 0.1;
                        player.move_vel.y = 0, player.force_vel.y = 0;
                    }
                }
            }
        }
    }
}

void Game::trace_collidables(in out GameTraceState trace_state, Ray ray) {
    collidable_scene.trace(trace_state, ray, 1);
    voxel_world.trace(trace_state, ray);
}

float Game::trace_depth(Ray ray, in uint max_steps) {
    GameTraceState trace_state;
    trace_state.default_init();
    trace_state.max_steps = max_steps;
    trace_collidables(trace_state, ray);
    // debug_scene.trace(trace_state, ray, 2);
    return trace_state.trace_record.dist;
}

float Game::draw_depth(uint2 pixel_i, float start_depth, in uint max_steps) {
    float2 view_uv = player.camera.create_view_uv(pixel_i);
    Ray ray = player.camera.create_view_ray(view_uv);
    ray.o += ray.nrm * start_depth;
    GameTraceRecord game_trace_record = trace(ray, max_steps);
    game_trace_record.trace_record.dist += start_depth;
    game_trace_record.draw_sample.depth += start_depth;
    return game_trace_record.trace_record.dist;
}

GameTraceRecord Game::trace(Ray ray, in uint max_steps) {
    GameTraceState trace_state;
    trace_state.default_init();
    trace_state.max_steps = max_steps;

    trace_collidables(trace_state, ray);
    debug_scene.trace(trace_state, ray, 2);

    trace_state.draw_sample.pos = ray.o + ray.nrm * trace_state.trace_record.dist;
    trace_state.draw_sample.nrm = trace_state.trace_record.nrm;
    trace_state.draw_sample.depth = trace_state.trace_record.dist;

    if (trace_state.trace_record.hit) {
        switch (trace_state.shape_type) {
        case 0:
            voxel_world.eval_color(trace_state);
            break;
        case 1:
            collidable_scene.eval_color(trace_state);
            break;
        case 2:
            debug_scene.eval_color(trace_state);
            break;
        }
    } else {
        trace_state.draw_sample.col = sample_sky(ray.nrm);
    }

    GameTraceRecord game_trace_record;
    game_trace_record.draw_sample = trace_state.draw_sample;
    game_trace_record.trace_record = trace_state.trace_record;
    return game_trace_record;
}

DrawSample Game::draw(in out Input input, uint2 pixel_i, float start_depth) {
    float uv_rand_offset = input.time;
    float2 uv_offset =
        float2(rand(float2(pixel_i) + uv_rand_offset + 10),
               rand(float2(pixel_i) + uv_rand_offset)) *
        0.0;

    float2 view_uv = player.camera.create_view_uv(pixel_i + uv_offset);
    Ray ray = player.camera.create_view_ray(view_uv);
    ray.o += ray.nrm * start_depth;

    GameTraceRecord game_trace_record = trace(ray, input.max_steps());
    game_trace_record.trace_record.dist += start_depth;
    game_trace_record.draw_sample.depth += start_depth;
    GameTraceRecord temp_trace_record;

    if (input.shadows_enabled() && game_trace_record.trace_record.hit) {
        ray.o = game_trace_record.draw_sample.pos + game_trace_record.draw_sample.nrm * 0.001;
        ray.nrm = sun_nrm;
        ray.inv_nrm = 1 / ray.nrm;
        temp_trace_record = trace(ray, input.max_steps());
        float3 shade = max(dot(game_trace_record.draw_sample.nrm, sun_nrm), 0) * sun_col;
        if (temp_trace_record.trace_record.hit) {
            shade = 0;
        }
        shade += sample_sky(game_trace_record.draw_sample.nrm) * 0.3;
        game_trace_record.draw_sample.col *= shade + 0.1;
    }

    float3 p = floor(game_trace_record.draw_sample.pos * VOXEL_SCL) / VOXEL_SCL;
    float p_dist = length(p - pick_pos[0]);
    if (p_dist < EDIT_RADIUS) {
        game_trace_record.draw_sample.col += float3(0.07, 0.08, 0.1);
    }

    // float3 chunk_p = floor(game_trace_record.draw_sample.pos * VOXEL_SCL / CHUNK_SIZE + 0.5) / VOXEL_SCL * CHUNK_SIZE;
    // float chunk_p_dist = length(chunk_p - pick_pos[0]) - CHUNK_SIZE / VOXEL_SCL;
    // if (chunk_p_dist < EDIT_RADIUS) {
    //     game_trace_record.draw_sample.col += float3(0.3, 0.08, 0.1);
    // }

    return game_trace_record.draw_sample;
}

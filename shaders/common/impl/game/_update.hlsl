#pragma once

#include "common/interface/game.hlsl"

#include "common/impl/game/_common.hlsl"
#include "common/impl/voxel_world/_update.hlsl"

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

    // debug_scene.add_shape(temp_sdf_world, float3(0, 0, 0), 2);

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

    sun_nrm = normalize(float3(-1.5, -1, 6));
    // editing = false;
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

    // debug_scene.sdf_worlds[0].origin = player.pos;
    // debug_scene.sdf_worlds[0].forward = debug_scene.capsules[0].forward;

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

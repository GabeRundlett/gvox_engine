#pragma once

#include "common/impl/raytrace.hlsl"
#include "common/impl/player.hlsl"

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

void Game::trace_collidables(in out GameTraceState trace_state, Ray ray) {
    collidable_scene.trace(trace_state, ray, 1);
    voxel_world.trace(trace_state, ray);
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

float Game::trace_depth(Ray ray, in uint max_steps) {
    GameTraceState trace_state;
    trace_state.default_init();
    trace_state.max_steps = max_steps;
    trace_collidables(trace_state, ray);
    // debug_scene.trace(trace_state, ray, 2);
    return trace_state.trace_record.dist;
}

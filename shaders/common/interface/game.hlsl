#pragma once

#include "common/interface/player.hlsl"
#include "common/interface/voxel_world.hlsl"

#define MAX_DEBUG_SPHERES 40
#define MAX_DEBUG_BOXES 10
#define MAX_DEBUG_CAPSULES 10
#define MAX_DEBUG_SDF_WORLDS 4
#define MAX_DEBUG_SHAPES (MAX_DEBUG_SPHERES + MAX_DEBUG_SPHERES + MAX_DEBUG_CAPSULES + MAX_DEBUG_SDF_WORLDS)

struct ShapeScene {
    Shape shapes[MAX_DEBUG_SHAPES];
    Sphere spheres[MAX_DEBUG_SPHERES];
    Box boxes[MAX_DEBUG_BOXES];
    Capsule capsules[MAX_DEBUG_CAPSULES];
    SdfWorld sdf_worlds[MAX_DEBUG_SDF_WORLDS];

    uint shape_n;
    uint sphere_n, box_n, capsule_n, sdf_world_n;

    void default_init() {
        shape_n = 0;
        sphere_n = 0, box_n = 0, capsule_n = 0, sdf_world_n = 0;
    }

    void add_shape(Sphere s, float3 color, uint material_id);
    void add_shape(Box b, float3 color, uint material_id);
    void add_shape(Capsule c, float3 color, uint material_id);
    void add_shape(SdfWorld s, float3 color, uint material_id);

    void trace(in out GameTraceState trace_state, Ray ray, int shape_type);
    void eval_color(in out GameTraceState trace_state);
};

struct GameTraceRecord {
    TraceRecord trace_record;
    DrawSample draw_sample;

    void default_init() {
        trace_record.default_init();
    }
};

struct Game {
    ShapeScene debug_scene;
    ShapeScene collidable_scene;
    Player player;
    float3 pick_pos[2];
    uint edit_state;
    float3 sun_nrm;

    VoxelWorld voxel_world;

    void default_init() {
        debug_scene.default_init();
        collidable_scene.default_init();
        player.default_init();
        voxel_world.default_init();
        edit_state = 0;
    }

    void init();
    void update(in out GpuInput input);
    void trace_collidables(in out GameTraceState trace_state, Ray ray);

    float trace_depth(Ray ray);
    float draw_depth(uint2 pixel_i, float start_depth);

    GameTraceRecord trace(Ray ray);
    DrawSample draw(in out GpuInput input, uint2 pixel_i, float start_depth);

    TraceRecord cube_min_trace(Ray ray, Box box);
};

#pragma once

#include "common/interface/shapes.hlsl"

#define TRACE_MAX_DIST 1e9

struct Ray {
    float3 o;
    float3 nrm, inv_nrm;

    void default_init() {
        o = float3(0, 0, 0);
        nrm = float3(1, 1, 1);
        inv_nrm = 1 / nrm;
    }
};

struct TraceRecord {
    bool hit;
    float dist;
    float3 nrm;

    void default_init() {
        hit = false;
        dist = TRACE_MAX_DIST;
        nrm = float3(0, 0, 0);
    }
};

struct SurfaceProps {
    uint material_type, object_type;
    float roughness, ior;

    void default_init() {
        material_type = 0;
        object_type = 0;
        roughness = 0;
        ior = 0;
    }
};

struct DrawSample {
    float3 col;
    float3 pos;
    float3 nrm;
    float depth;
    float lifetime;
};

struct GameTraceState {
    TraceRecord trace_record;
    DrawSample draw_sample;
    int shape_type;
    int shape_i;
    uint max_steps;

    void default_init() {
        trace_record.default_init();
        shape_type = 0, shape_i = 0;
        max_steps = 10;
    }
};

TraceRecord trace_sphere(Ray ray, Sphere s);
TraceRecord trace_box(Ray ray, Box s);

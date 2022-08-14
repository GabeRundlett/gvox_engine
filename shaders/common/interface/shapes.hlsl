#pragma once

enum class ShapeType {
    Invalid,
    Sphere,
    Box,
    Capsule,
};

struct Shape {
    ShapeType type;
    uint storage_index;
    uint material_id;

    float3 color;

    void default_init() {
        type = ShapeType::Invalid;
        storage_index = 0;
        material_id = 0;
    }
};

struct Sphere {
    float3 o;
    float r;

    void default_init() {
        o = float3(0, 0, 0);
        r = 1;
    }

    bool inside(float3 p);
};

struct Capsule {
    float3 p0, p1;
    float2 forward;
    float r;

    void default_init() {
        p0 = float3(0, 0, 0);
        p1 = float3(0, 0, 0);
        r = 1;
    }

    bool inside(float3 p);
};

struct Box {
    float3 bound_min, bound_max;

    void default_init() {
        bound_min = float3(0, 0, 0);
        bound_max = float3(1, 1, 1);
    }

    bool inside(float3 p);
};

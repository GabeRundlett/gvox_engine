#pragma once

#include "common/interface/shapes.hlsl"

bool Sphere::inside(float3 p) {
    return dot(p - o, p - o) < r * r;
}

bool Box::inside(float3 p) {
    return all(p >= bound_min) && all(p <= bound_max);
}

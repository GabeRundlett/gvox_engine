#pragma once

#include "common/interface/raytrace.hlsl"
#include "common/impl/shapes.hlsl"

TraceRecord trace_sphere(Ray ray, Sphere s) {
    TraceRecord result;
    result.default_init();

    float3 so_r = ray.o - s.o;
    float a = dot(ray.nrm, ray.nrm);
    float b = 2.0f * dot(ray.nrm, so_r);
    float c = dot(so_r, so_r) - (s.r * s.r);
    float f = b * b - 4.0f * a * c;
    if (f < 0.0f)
        return result;
    result.dist = (-b - sqrt(f)) / (2.0f * a);
    result.hit = result.dist > 0.0f;
    result.nrm = normalize(ray.o + ray.nrm * result.dist - s.o);
    return result;
}

TraceRecord trace_box(Ray ray, Box b) {
    TraceRecord result;
    result.default_init();
    if (b.inside(ray.o)) {
        result.hit = true;
        result.dist = 0;
        result.nrm = 0;
    }

    float tx1 = (b.bound_min.x - ray.o.x) * ray.inv_nrm.x;
    float tx2 = (b.bound_max.x - ray.o.x) * ray.inv_nrm.x;
    float tmin = min(tx1, tx2);
    float tmax = max(tx1, tx2);
    float ty1 = (b.bound_min.y - ray.o.y) * ray.inv_nrm.y;
    float ty2 = (b.bound_max.y - ray.o.y) * ray.inv_nrm.y;
    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));
    float tz1 = (b.bound_min.z - ray.o.z) * ray.inv_nrm.z;
    float tz2 = (b.bound_max.z - ray.o.z) * ray.inv_nrm.z;
    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));

    result.hit = (tmax >= tmin && tmin > 0);
    result.dist = tmin;

    bool is_x = tmin == tx1 || tmin == tx2;
    bool is_y = tmin == ty1 || tmin == ty2;
    bool is_z = tmin == tz1 || tmin == tz2;

    if (is_z) {
        if (ray.nrm.z < 0) {
            result.nrm = float3(0, 0, 1);
        } else {
            result.nrm = float3(0, 0, -1);
        }
    } else if (is_y) {
        if (ray.nrm.y < 0) {
            result.nrm = float3(0, 1, 0);
        } else {
            result.nrm = float3(0, -1, 0);
        }
    } else {
        if (ray.nrm.x < 0) {
            result.nrm = float3(1, 0, 0);
        } else {
            result.nrm = float3(-1, 0, 0);
        }
    }

    return result;
}

TraceRecord trace_capsule(Ray ray, Capsule cap) {
    TraceRecord result;
    result.default_init();
    float3 ba = cap.p1 - cap.p0;
    float3 oa = ray.o - cap.p0;
    float baba = dot(ba, ba);
    float bard = dot(ba, ray.nrm);
    float baoa = dot(ba, oa);
    float rdoa = dot(ray.nrm, oa);
    float oaoa = dot(oa, oa);
    float a = baba - bard * bard;
    float b = baba * rdoa - baoa * bard;
    float c = baba * oaoa - baoa * baoa - cap.r * cap.r * baba;
    float h = b * b - a * c;
    if (h >= 0.) {
        float t = (-b - sqrt(h)) / a;
        float d = TRACE_MAX_DIST;
        float y = baoa + t * bard;
        if (y > 0. && y < baba) {
            d = t;
        } else {
            float3 oc = (y <= 0.) ? oa : ray.o - cap.p1;
            b = dot(ray.nrm, oc);
            c = dot(oc, oc) - cap.r * cap.r;
            h = b * b - c;
            if (h > 0.0) {
                d = -b - sqrt(h);
            }
        }
        // if (d >= distBound.x && d <= distBound.y) {
        cap.p0 = ray.o + ray.nrm * d - cap.p0;
        float h = clamp(dot(cap.p0, ba) / dot(ba, ba), 0.0, 1.0);
        result.nrm = (cap.p0 - h * ba) / cap.r;
        result.dist = d;
        result.hit = d != TRACE_MAX_DIST && d >= 0;
        // return d;
        // }
    }
    return result;
}

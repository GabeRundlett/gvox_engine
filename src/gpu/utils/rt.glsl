#pragma once

#include "math_const.glsl"
#include "gbuffer.glsl"
#include "ray_cone.glsl"

struct GbufferRayPayload {
    GbufferDataPacked gbuffer_packed;
    float t;
    RayCone ray_cone;
    uint path_length;
};

GbufferRayPayload GbufferRayPayload_new_miss() {
    GbufferRayPayload res;
    res.t = FLT_MAX;
    res.ray_cone = RayCone_from_spread_angle(0.0);
    res.path_length = 0;
    return res;
}

bool is_miss(inout GbufferRayPayload self) { return self.t == FLT_MAX; }
bool is_hit(inout GbufferRayPayload self) { return !is_miss(self); }

struct ShadowRayPayload {
    bool is_shadowed;
};

ShadowRayPayload ShadowRayPayload_new_hit() {
    ShadowRayPayload res;
    res.is_shadowed = true;
    return res;
}

bool is_miss(inout ShadowRayPayload self) { return !self.is_shadowed; }
bool is_hit(inout ShadowRayPayload self) { return !is_miss(self); }

struct RayDesc {
    daxa_f32vec3 Origin;
    daxa_f32 TMin;
    daxa_f32vec3 Direction;
    daxa_f32 TMax;
};
RayDesc new_ray(vec3 origin, vec3 direction, float tmin, float tmax) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tmin;
    ray.TMax = tmax;
    return ray;
}

struct GbufferPathVertex {
    bool is_hit;
    GbufferDataPacked gbuffer_packed;
    vec3 position;
    float ray_t;
};

struct GbufferRaytrace {
    RayDesc ray;
    RayCone ray_cone;
    uint path_length;
    bool cull_back_faces;
};

GbufferRaytrace GbufferRaytrace_with_ray(RayDesc ray) {
    GbufferRaytrace res;
    res.ray = ray;
    res.ray_cone = RayCone_from_spread_angle(1.0);
    res.path_length = 0;
    res.cull_back_faces = true;
    return res;
}

GbufferRaytrace with_cone(inout GbufferRaytrace self, RayCone ray_cone) {
    GbufferRaytrace res = self;
    res.ray_cone = ray_cone;
    return res;
}

GbufferRaytrace with_path_length(inout GbufferRaytrace self, uint v) {
    GbufferRaytrace res = self;
    res.path_length = v;
    return res;
}

GbufferRaytrace with_cull_back_faces(inout GbufferRaytrace self, bool v) {
    GbufferRaytrace res = self;
    res.cull_back_faces = v;
    return res;
}

GbufferPathVertex trace(GbufferRaytrace self, VoxelBufferPtrs voxels_buffer_ptrs) {
    GbufferRayPayload payload = GbufferRayPayload_new_miss();
    payload.ray_cone = self.ray_cone;
    payload.path_length = self.path_length;

    // uint trace_flags = 0;
    // if (self.cull_back_faces) {
    //     trace_flags |= RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    // }
    // TraceRay(acceleration_structure, trace_flags, 0xff, 0, 0, 0, self.ray, payload);

    VoxelTraceResult trace_result = voxel_trace(VoxelTraceInfo(voxels_buffer_ptrs, self.ray.Direction, MAX_STEPS, self.ray.TMax, self.ray.TMin, true), self.ray.Origin);
    const bool is_hit = trace_result.dist < self.ray.TMax;

    if (is_hit) {
        GbufferPathVertex res;
        res.is_hit = true;
        res.position = self.ray.Origin;
        res.gbuffer_packed = payload.gbuffer_packed;
        res.ray_t = trace_result.dist;
        return res;
    } else {
        GbufferPathVertex res;
        res.is_hit = false;
        res.ray_t = FLT_MAX;
        return res;
    }
}

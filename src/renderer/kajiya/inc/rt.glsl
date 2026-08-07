#pragma once

#include <renderer/kajiya/inc/math_const.glsl>
#include <renderer/kajiya/inc/gbuffer.glsl>
#include <renderer/kajiya/inc/ray_cone.glsl>

#include <renderer/rt.glsl>

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
    vec3 Origin;
    float TMin;
    vec3 Direction;
    float TMax;
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

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_RAYGEN
#include <utilities/gpu/normal.glsl>

GbufferPathVertex trace(GbufferRaytrace self) {
    const uint ray_flags = gl_RayFlagsNoneEXT;
    const uint cull_mask = 0xFF;
    const uint sbt_record_offset = 0;
    const uint sbt_record_stride = 0;
    const uint miss_index = 0;

    traceRayEXT(
        accelerationStructureEXT(push.uses.tlas),
        ray_flags, cull_mask, sbt_record_offset, sbt_record_stride, miss_index,
        self.ray.Origin, self.ray.TMin, self.ray.Direction, self.ray.TMax, PAYLOAD_LOC);

    if (prd.data1 != miss_ray_payload().data1) {
        vec3 world_pos = vec3(0);
        vec3 _unused_vel = vec3(0);
        // PackedVoxel voxel_data = unpack_ray_payload(push.uses.geometry_pointers, push.uses.attribute_pointers, push.uses.blas_transforms, prd, Ray(self.ray.Origin, self.ray.Direction), world_pos, _unused_vel);
        // Voxel voxel = unpack_voxel(voxel_data);
        GpuVoxel voxel;
        voxel.albedo = vec3(0.5);
        voxel.normal = vec3(0,0,1);
        voxel.roughness = 1;
        voxel.material_type = 1;


#if PER_VOXEL_NORMALS
        vec3 ws_nrm = voxel.normal;
#else
        vec3 ws_nrm = voxel_face_normal((floor(world_pos * VOXEL_SCL + self.ray.Direction * 0.001) + 0.5) * VOXEL_SIZE, Ray(self.ray.Origin, self.ray.Direction), vec3(1.0) / self.ray.Direction);
#endif

        GbufferPathVertex res;
        res.is_hit = true;
        res.position = world_pos;
        res.gbuffer_packed.data0 = uvec4(0);
        res.gbuffer_packed.data0.x = voxel_data.data;
        res.gbuffer_packed.data0.y = nrm_to_u16(ws_nrm);
        res.ray_t = length(world_pos - self.ray.Origin);
        return res;
    } else {
        GbufferPathVertex res;
        res.is_hit = false;
        res.ray_t = FLT_MAX;
        return res;
    }
}
#endif

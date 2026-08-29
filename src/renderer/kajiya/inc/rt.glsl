#ifndef RENDERER_KAJIYA_INC_RT_GLSL
#define RENDERER_KAJIYA_INC_RT_GLSL

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
#include <voxels/pack_unpack.inl>

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
        vec3 world_pos = self.ray.Origin + self.ray.Direction * prd.t;
        vec3 _unused_vel = vec3(0);
        Voxel voxel = unpack_ray_payload(prd, push.uses.voxel_object_manifests);

        GbufferPathVertex res;
        res.is_hit = true;
        res.position = world_pos;
        res.gbuffer_packed.data0 = uvec4(0);
        res.gbuffer_packed.data0.x = pack_voxel(voxel).data;
        res.gbuffer_packed.data0.y = nrm_to_u16(voxel.normal);
        res.ray_t = length(world_pos - self.ray.Origin);
        return res;
    } else {
        GbufferPathVertex res;
        res.is_hit = false;
        res.ray_t = FLT_MAX;
        return res;
    }
}

bool rt_is_shadowed(RayDesc ray) {
    ShadowRayPayload shadow_payload = ShadowRayPayload_new_hit();
    const uint ray_flags = gl_RayFlagsTerminateOnFirstHitEXT;
    const uint cull_mask = 0xFF;
    const uint sbt_record_offset = 0;
    const uint sbt_record_stride = 0;
    const uint miss_index = 0;
    traceRayEXT(
        accelerationStructureEXT(push.uses.tlas),
        ray_flags, cull_mask, sbt_record_offset, sbt_record_stride, miss_index,
        ray.Origin, ray.TMin, ray.Direction, ray.TMax, PAYLOAD_LOC);
    shadow_payload.is_shadowed = prd.data1 != miss_ray_payload().data1;
    return shadow_payload.is_shadowed;
}

#elif GL_COMPUTE_SHADER && TRACE

#include <utilities/gpu/normal.glsl>
#include <voxels/pack_unpack.inl>

GbufferPathVertex trace(GbufferRaytrace self) {
    const uint ray_flags = gl_RayFlagsNoneEXT;
    const uint cull_mask = 0xFF;
    const uint miss_index = 0;
    VoxelHit hit;

    VoxelHit nearest_hit_attrib = voxel_miss_hit_attrib();
    rayQueryInitializeEXT(
        ray_query, accelerationStructureEXT(push.uses.tlas),
        ray_flags, cull_mask, self.ray.Origin, self.ray.TMin, self.ray.Direction, self.ray.TMax);

    prd = miss_ray_payload();

    while (rayQueryProceedEXT(ray_query)) {
        uint type = rayQueryGetIntersectionTypeEXT(ray_query, false);
        if (type == gl_RayQueryCandidateIntersectionAABBEXT) {
            const float t_aabb = rayQueryGetIntersectionTEXT(ray_query, false);
            if (t_aabb < self.ray.TMax && t_aabb < rayQueryGetIntersectionTEXT(ray_query, true)) {

                uint instanceIndex = rayQueryGetIntersectionInstanceIdEXT(ray_query, false);
                uint primitiveIndex = rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, false);

                daxa_BufferPtr(GpuVoxelObject) voxel_object = advance(push.uses.voxel_object_manifests, instanceIndex);
                daxa_BufferPtr(BrickPrimitive) brick_primitive_ptr = advance(deref(voxel_object).brick_primitives, primitiveIndex);
                const vec3 voxelOffset = vec3(deref(brick_primitive_ptr).offset & BRICK_MASK);
                const vec3 brickOffset = vec3(deref(brick_primitive_ptr).offset & ~BRICK_MASK);
                vec3 size = vec3(float(deref(brick_primitive_ptr).size_x), float(deref(brick_primitive_ptr).size_y), float(deref(brick_primitive_ptr).size_z));

                vec3 localOrig = rayQueryGetIntersectionObjectRayOriginEXT(ray_query, false) - brickOffset;
                vec3 localDir = rayQueryGetIntersectionObjectRayDirectionEXT(ray_query, false);

                vec2 t = RayAabbIntersect(localOrig - voxelOffset, localDir, size);
                t.x = max(0.0, t.x);

                if (t.x < t.y) {
                    vec3 o = localOrig + localDir * t.x;
                    ivec3 hitCoord;
                    int nrm = RayAabbIntersectNormal(localOrig - voxelOffset, localDir, size);
                    float hitDist = traceVoxelDataBitmap(brick_primitive_ptr, o, localDir, hitCoord, nrm);
                    if (hitDist >= 0.0 && (t.x + hitDist) < rayQueryGetIntersectionTEXT(ray_query, true)) {
                        float dist = t.x + hitDist;
                        hit.x = uint8_t(hitCoord.x);
                        hit.y = uint8_t(hitCoord.y);
                        hit.z = uint8_t(hitCoord.z);
                        hit.nrm = uint8_t(nrm);
                        rayQueryGenerateIntersectionEXT(ray_query, dist);
                    }
                }
            }
        }
    }
    if (rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionGeneratedEXT) {
        uint instance_custom_index = rayQueryGetIntersectionInstanceCustomIndexEXT(ray_query, true);
        uint prim_index = rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true);
        prd = pack_ray_payload(instance_custom_index, prim_index, rayQueryGetIntersectionTEXT(ray_query, true), hit);
    }

    if (prd.data1 != miss_ray_payload().data1) {
        vec3 world_pos = self.ray.Origin + self.ray.Direction * prd.t;
        vec3 _unused_vel = vec3(0);
        Voxel voxel = unpack_ray_payload(prd, push.uses.voxel_object_manifests);

        GbufferPathVertex res;
        res.is_hit = true;
        res.position = world_pos;
        res.gbuffer_packed.data0 = uvec4(0);
        res.gbuffer_packed.data0.x = pack_voxel(voxel).data;
        res.gbuffer_packed.data0.y = nrm_to_u16(voxel.normal);
        res.ray_t = rayQueryGetIntersectionTEXT(ray_query, true);
        return res;
    } else {
        GbufferPathVertex res;
        res.is_hit = false;
        res.ray_t = FLT_MAX;
        return res;
    }
}

bool rt_is_shadowed(RayDesc ray) {
    const uint ray_flags = gl_RayFlagsTerminateOnFirstHitEXT;
    const uint cull_mask = 0xFF;
    const uint miss_index = 0;

    rayQueryInitializeEXT(
        ray_query, accelerationStructureEXT(push.uses.tlas),
        ray_flags, cull_mask, ray.Origin, ray.TMin, ray.Direction, ray.TMax);

    while (rayQueryProceedEXT(ray_query)) {
        uint type = rayQueryGetIntersectionTypeEXT(ray_query, false);
        if (type == gl_RayQueryCandidateIntersectionAABBEXT) {
            const float t_aabb = rayQueryGetIntersectionTEXT(ray_query, false);
            if (t_aabb < ray.TMax && t_aabb < rayQueryGetIntersectionTEXT(ray_query, true)) {

                uint instanceIndex = rayQueryGetIntersectionInstanceIdEXT(ray_query, false);
                uint primitiveIndex = rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, false);

                daxa_BufferPtr(GpuVoxelObject) voxel_object = advance(push.uses.voxel_object_manifests, instanceIndex);
                daxa_BufferPtr(BrickPrimitive) brick_primitive_ptr = advance(deref(voxel_object).brick_primitives, primitiveIndex);
                const vec3 voxelOffset = vec3(deref(brick_primitive_ptr).offset & BRICK_MASK);
                const vec3 brickOffset = vec3(deref(brick_primitive_ptr).offset & ~BRICK_MASK);
                vec3 size = vec3(float(deref(brick_primitive_ptr).size_x), float(deref(brick_primitive_ptr).size_y), float(deref(brick_primitive_ptr).size_z));

                vec3 localOrig = rayQueryGetIntersectionObjectRayOriginEXT(ray_query, false) - brickOffset;
                vec3 localDir = rayQueryGetIntersectionObjectRayDirectionEXT(ray_query, false);

                vec2 t = RayAabbIntersect(localOrig - voxelOffset, localDir, size);
                t.x = max(0.0, t.x);

                if (t.x < t.y) {
                    vec3 o = localOrig + localDir * t.x;
                    ivec3 hitCoord;
                    int nrm = RayAabbIntersectNormal(localOrig - voxelOffset, localDir, size);
                    float hitDist = traceVoxelDataBitmap(brick_primitive_ptr, o, localDir, hitCoord, nrm);
                    if (hitDist >= 0.0) {
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

#endif

#endif // RENDERER_KAJIYA_INC_RT_GLSL

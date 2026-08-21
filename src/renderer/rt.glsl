#pragma once

#include <voxels/voxel.glsl>
#include <application/input.inl>

#define PAYLOAD_LOC 0

struct RayPayload {
    uint data0;
    uint data1;
    float t;
};

struct Ray {
    daxa_f32vec3 origin;
    daxa_f32vec3 direction;
};

#include <voxels/pack_unpack.inl>

RayPayload pack_ray_payload(uint blas_id, uint brick_id, float ray_t, VoxelHit hit) {
    RayPayload result;
    result.data0 = blas_id;
    result.data1 = (brick_id << 11) | (hit.x << 0) | (uint(hit.y) << 3) | (uint(hit.z) << 6) | (uint(hit.nrm) << 9);
    result.t = ray_t;
    return result;
}

Voxel unpack_ray_payload(RayPayload payload, daxa_BufferPtr(GpuVoxelObject) voxel_object_manifests) {
    uint blas_id = payload.data0;
    uint brick_id = payload.data1 >> 11;

    uvec3 voxel_i;
    voxel_i.x = (payload.data1 >> 0) & 0x7;
    voxel_i.y = (payload.data1 >> 3) & 0x7;
    voxel_i.z = (payload.data1 >> 6) & 0x7;
    uint nrm = (payload.data1 >> 9) & 0x3;

    daxa_BufferPtr(GpuVoxelObject) voxel_object = advance(voxel_object_manifests, blas_id);
    daxa_BufferPtr(VoxelShadingAttribBrick) shading_brick = advance(deref(voxel_object).brick_shading_attribs, brick_id);
    uint voxel_index = voxel_i.x + voxel_i.y * BRICK_SIZE + voxel_i.z * BRICK_SIZE * BRICK_SIZE;
    PackedVoxel packed_voxel = deref(shading_brick).voxels[voxel_index];

    Voxel voxel = unpack_voxel(packed_voxel);
    voxel.albedo *= deref(voxel_object).tint;

    // switch (nrm)
    // {
    // case 0: voxel.normal = -sign(ray_d) * vec3(1, 0, 0); break;
    // case 1: voxel.normal = -sign(ray_d) * vec3(0, 1, 0); break;
    // case 2: voxel.normal = -sign(ray_d) * vec3(0, 0, 1); break;
    // }

    return voxel;
}

VoxelHit voxel_miss_hit_attrib() {
    return VoxelHit(uint8_t(BRICK_SIZE), uint8_t(BRICK_SIZE), uint8_t(BRICK_SIZE), uint8_t(0));
}

RayPayload miss_ray_payload() {
    return RayPayload(0, 0, -1);
}

// Ray-AABB intersection
vec2 ray_aabb(vec3 rayOrigin, vec3 rayDir, vec3 size) {
    vec3 tMin = -rayOrigin / rayDir;
    vec3 tMax = (size - rayOrigin) / rayDir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return vec2(tNear, tFar);
}
int ray_aabb_normal(vec3 rayOrigin, vec3 rayDir, vec3 size) {
    vec3 tMin = -rayOrigin / rayDir;
    vec3 tMax = (size - rayOrigin) / rayDir;
    vec3 t1 = min(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    vec3 normal;
    if (tNear == t1.x)
        return 0;
    else if (tNear == t1.y)
        return 1;
    else
        return 2;
}

float hitAabb(const Aabb aabb, const Ray r) {
    if (all(greaterThanEqual(r.origin, aabb.min)) && all(lessThanEqual(r.origin, aabb.max))) {
        return 0.0;
    }
    vec3 invDir = 1.0 / r.direction;
    vec3 tbot = invDir * (aabb.min - r.origin);
    vec3 ttop = invDir * (aabb.max - r.origin);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    float t0 = max(tmin.x, max(tmin.y, tmin.z));
    float t1 = min(tmax.x, min(tmax.y, tmax.z));
    return t1 > max(t0, 0.0) ? t0 : -1.0;
}

float hitAabb_midpoint(const Aabb aabb, const Ray r) {
    vec3 invDir = 1.0 / r.direction;
    vec3 tbot = invDir * (aabb.min - r.origin);
    vec3 ttop = invDir * (aabb.max - r.origin);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    float t0 = max(tmin.x, max(tmin.y, tmin.z));
    float t1 = min(tmax.x, min(tmax.y, tmax.z));
    return (t0 + t1) * 0.5;
}

vec3 voxel_face_normal(vec3 center, Ray ray, in vec3 _invRayDir) {
    ray.origin = ray.origin - center;
    float winding = 1;
    vec3 sgn = -sign(ray.direction);
    // Distance to plane
    vec3 d = VOXEL_SIZE * 0.5 * winding * sgn - ray.origin;
    d *= _invRayDir;
#define TEST(U, VW) (d.U >= 0.0) && all(lessThan(abs(ray.origin.VW + ray.direction.VW * d.U), vec2(VOXEL_SIZE * 0.5)))
    bvec3 test = bvec3(TEST(x, yz), TEST(y, zx), TEST(z, xy));
    sgn = test.x ? vec3(sgn.x, 0, 0) : (test.y ? vec3(0, sgn.y, 0) : vec3(0, 0, test.z ? sgn.z : 0));
#undef TEST
    return sgn;
}

vec2 RayAabbIntersect(vec3 rayOrigin, vec3 rayDir, vec3 size) {
    vec3 tMin = -rayOrigin / rayDir;
    vec3 tMax = (size - rayOrigin) / rayDir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return vec2(tNear, tFar);
}

int RayAabbIntersectNormal(vec3 rayOrigin, vec3 rayDir, vec3 size) {
    vec3 tMin = -rayOrigin / rayDir;
    vec3 tMax = (size - rayOrigin) / rayDir;
    vec3 t1 = min(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    vec3 normal;
    if (tNear == t1.x)
        return 0;
    else if (tNear == t1.y)
        return 1;
    else
        return 2;
}

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_INTERSECTION || DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_CLOSEST_HIT
#else
#extension GL_EXT_ray_query : enable
rayQueryEXT ray_query;
#endif

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_INTERSECTION
hitAttributeEXT VoxelHit hit;

void main() {
    daxa_BufferPtr(GpuVoxelObject) voxelObject = advance(push.uses.voxel_object_manifests, gl_InstanceID);
    daxa_BufferPtr(BrickPrimitive) brickPrimitivePtr = advance(deref(voxelObject).brick_primitives, gl_PrimitiveID);
    const float scale = 1; // deref(brickPrimitivePtr).scale;
    const vec3 voxelOffset = vec3(deref(brickPrimitivePtr).offset & BRICK_MASK) * scale;
    const vec3 brickOffset = vec3(deref(brickPrimitivePtr).offset & ~BRICK_MASK) * scale;
    vec3 size = vec3(float(deref(brickPrimitivePtr).size_x), float(deref(brickPrimitivePtr).size_y), float(deref(brickPrimitivePtr).size_z)) * scale;

    vec3 localOrig = gl_ObjectRayOriginEXT - brickOffset;
    vec3 localDir = gl_ObjectRayDirectionEXT;

    vec2 t = RayAabbIntersect(localOrig - voxelOffset, localDir, size);
    t.x = max(0.0, t.x);

    if (t.x < t.y) {
        vec3 o = localOrig + localDir * t.x;
        ivec3 hitCoord;
        int nrm = RayAabbIntersectNormal(localOrig - voxelOffset, localDir, size);
        float hitDist = traceVoxelDataBitmap(brickPrimitivePtr, o, localDir, hitCoord, nrm);
        if (hitDist >= 0.0) {
            float dist = t.x + hitDist;
            hit.x = uint8_t(hitCoord.x);
            hit.y = uint8_t(hitCoord.y);
            hit.z = uint8_t(hitCoord.z);
            hit.nrm = uint8_t(nrm);
            reportIntersectionEXT(dist, 0);
        }
    }
}
#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_CLOSEST_HIT
hitAttributeEXT VoxelHit hit;

layout(location = PAYLOAD_LOC) rayPayloadInEXT RayPayload prd;
void main() {
    prd = pack_ray_payload(gl_InstanceCustomIndexEXT, gl_PrimitiveID, gl_HitTEXT, hit);
}
#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_MISS
layout(location = PAYLOAD_LOC) rayPayloadInEXT RayPayload prd;
void main() {
    prd = miss_ray_payload();
}
#endif

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_COMPUTE
struct VoxelRtTraceInfo {
    VoxelRtBufferPtrs ptrs;
    vec3 ray_dir;
    float max_dist;
};

struct VoxelTraceResult {
    float dist;
    vec3 nrm;
    vec3 vel;
    uint step_n;
    // PackedVoxel voxel_data;
};

VoxelTraceResult voxel_trace(in VoxelRtTraceInfo info, in out vec3 ray_pos) {
    VoxelTraceResult result;

    const uint ray_flags = gl_RayFlagsNoOpaqueEXT;
    const uint cull_mask = 0xFF & ~(0x01);
    // const uint cull_mask = 0xFF;
    const uint sbt_record_offset = 0;
    const uint sbt_record_stride = 0;
    const uint miss_index = 0;
    const float t_min = 0.001;
    const float t_max = 10000;
    VoxelHit nearest_hit_attrib = voxel_miss_hit_attrib();
    rayQueryInitializeEXT(
        ray_query, accelerationStructureEXT(info.ptrs.tlas),
        ray_flags, cull_mask, ray_pos, t_min, info.ray_dir, t_max);
    while (rayQueryProceedEXT(ray_query)) {
        uint type = rayQueryGetIntersectionTypeEXT(ray_query, false);
        if (type == gl_RayQueryCandidateIntersectionAABBEXT) {
            const float t_aabb = rayQueryGetIntersectionTEXT(ray_query, false);
            if (t_aabb < t_max && t_aabb < rayQueryGetIntersectionTEXT(ray_query, true)) {
                // intersect_voxel_brick(info.ptrs.geometry_pointers);
            }
        }
    }
    result.dist = rayQueryGetIntersectionTEXT(ray_query, true);
    if (rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionGeneratedEXT) {
        uint instance_custom_index = rayQueryGetIntersectionInstanceCustomIndexEXT(ray_query, true);
        uint prim_index = rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true);
        RayPayload prd = pack_ray_payload(instance_custom_index, prim_index, 0, nearest_hit_attrib);
        // result.voxel_data = unpack_ray_payload(info.ptrs.geometry_pointers, info.ptrs.attribute_pointers, info.ptrs.blas_transforms, prd, Ray(ray_pos, info.ray_dir), ray_pos, result.vel);
        // result.voxel_data = 0xffffffff;
        // Voxel voxel = unpack_voxel(result.voxel_data);
        // result.nrm = voxel.normal;
    }
    return result;
}
#endif

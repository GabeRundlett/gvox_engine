#pragma once

#include <voxels/voxel.glsl>
#include <application/input.inl>

#define PAYLOAD_LOC 0

struct RayPayload {
    uint data0;
    uint data1;
};

struct Ray {
    daxa_f32vec3 origin;
    daxa_f32vec3 direction;
};

RayPayload pack_ray_payload(uint blas_id, uint brick_id, VoxelHit hit) {
    return RayPayload(blas_id, (brick_id * (BRICK_SIZE * BRICK_SIZE * BRICK_SIZE * 2)) |
                                   ((hit.x << 0) | (uint(hit.y) << 8) | (uint(hit.z) << 16) | (uint(hit.nrm) << 24)));
}

VoxelHit voxel_miss_hit_attrib() {
    return VoxelHit(uint8_t(BRICK_SIZE), uint8_t(BRICK_SIZE), uint8_t(BRICK_SIZE), uint8_t(0));
}

RayPayload miss_ray_payload() {
    return pack_ray_payload(0, 0, voxel_miss_hit_attrib());
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

// PackedVoxel unpack_ray_payload(
//     daxa_BufferPtr(daxa_BufferPtr(BlasGeom)) geometry_pointers,
//     daxa_BufferPtr(daxa_BufferPtr(VoxelBrickAttribs)) attribute_pointers,
//     daxa_BufferPtr(VoxelBlasTransform) blas_transforms,
//     RayPayload payload, Ray ray, out vec3 hit_pos, out vec3 hit_vel) {
//     uint blas_id = payload.data0;
//     uint brick_id = payload.data1 / (BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * 2);
//     uint voxel_index = payload.data1 & (BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * 2 - 1);
//     daxa_BufferPtr(VoxelBrickAttribs) brick_attribs = deref(advance(attribute_pointers, blas_id));
//     daxa_BufferPtr(BlasGeom) blas_geoms = deref(advance(geometry_pointers, blas_id));
//     {
//         // mat3x4 m = deref(advance(blas_transforms, blas_id));
//         // mat4 world_to_blas = mat4(m[0], m[1], m[2], vec4(0, 0, 0, 1));
//         // mat4 blas_to_world = transpose(world_to_blas);
//         hit_vel = deref(advance(blas_transforms, blas_id)).vel;
//         vec3 v = deref(advance(blas_transforms, blas_id)).pos;
//         Aabb aabb = deref(advance(blas_geoms, brick_id)).aabb;
//         ivec3 mapPos = ivec3(voxel_index % BLAS_BRICK_SIZE, (voxel_index / BLAS_BRICK_SIZE) % BLAS_BRICK_SIZE, voxel_index / BLAS_BRICK_SIZE / BLAS_BRICK_SIZE);
//         aabb.minimum = vec3(ivec3(floor(aabb.minimum * VOXEL_SCL)) & ~0x7) * VOXEL_SIZE;
//         aabb.minimum += vec3(mapPos) * VOXEL_SIZE;
//         aabb.maximum = aabb.minimum + VOXEL_SIZE;
//         ray.origin -= v;
//         hit_pos = ray.origin + ray.direction * hitAabb(aabb, ray);
//         hit_pos += v;
//         // hit_pos = (blas_to_world * vec4(hit_pos, 1)).xyz;
//     }
//     return deref(advance(brick_attribs, brick_id)).packed_voxels[voxel_index];
// }

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_INTERSECTION || DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_CLOSEST_HIT
// hitAttributeEXT HitAttribute hit_attrib;
#else
#extension GL_EXT_ray_query : enable
rayQueryEXT ray_query;
// HitAttribute hit_attrib;
#endif

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_INTERSECTION
hitAttributeEXT VoxelHit hit;

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

void main() {
    // return;
    daxa_BufferPtr(ChunkPrimitive) chunkPrimitivePtr = deref(advance(push.uses.chunk_primitive_pointers, gl_PrimitiveID));
    const float scale = 1; // deref(chunkPrimitivePtr).scale;
    const vec3 voxelOffset = vec3(deref(chunkPrimitivePtr).offset & CHUNK_MASK) * scale;
    const vec3 chunkOffset = vec3(deref(chunkPrimitivePtr).offset & ~CHUNK_MASK) * scale;
    vec3 size = vec3(float(deref(chunkPrimitivePtr).size_x), float(deref(chunkPrimitivePtr).size_y), float(deref(chunkPrimitivePtr).size_y)) * scale;

    vec3 localOrig = gl_ObjectRayOriginEXT - chunkOffset;
    vec3 localDir = gl_ObjectRayDirectionEXT;

    vec2 t = RayAabbIntersect(localOrig - voxelOffset, localDir, size);
    t.x = max(0.0, t.x);

    if (t.x < t.y) {
        reportIntersectionEXT(t.x, 0);
        // vec3 o = localOrig + localDir * t.x;
        // ivec3 hitCoord;
        // int nrm = RayAabbIntersectNormal(localOrig - voxelOffset, localDir, size);
        // float hitDist = traceVoxelDataBitmap(chunkPrimitivePtr, o, localDir, hitCoord, nrm);
        // if (hitDist >= 0.0) {
        //     float dist = t.x + hitDist;
        //     hit.x = uint8_t(hitCoord.x);
        //     hit.y = uint8_t(hitCoord.y);
        //     hit.z = uint8_t(hitCoord.z);
        //     hit.nrm = uint8_t(nrm);
        //     reportIntersectionEXT(dist, 0);
        // }
    }
}
#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_CLOSEST_HIT
hitAttributeEXT VoxelHit hit;

layout(location = PAYLOAD_LOC) rayPayloadInEXT RayPayload prd;
void main() {
    prd = pack_ray_payload(gl_InstanceCustomIndexEXT, gl_PrimitiveID, hit);
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
        RayPayload prd = pack_ray_payload(instance_custom_index, prim_index, nearest_hit_attrib);
        // result.voxel_data = unpack_ray_payload(info.ptrs.geometry_pointers, info.ptrs.attribute_pointers, info.ptrs.blas_transforms, prd, Ray(ray_pos, info.ray_dir), ray_pos, result.vel);
        // result.voxel_data = 0xffffffff;
        // Voxel voxel = unpack_voxel(result.voxel_data);
        // result.nrm = voxel.normal;
    }
    return result;
}
#endif

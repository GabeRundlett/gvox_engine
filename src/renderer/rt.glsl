#pragma once

#include <voxels/voxels.glsl>
#include <application/input.inl>
#include <voxels/voxels.inl>

#define PAYLOAD_LOC 0

struct HitAttribute {
    uint data;
};

struct RayPayload {
    uint data0;
    uint data1;
};

struct Ray {
    daxa_f32vec3 origin;
    daxa_f32vec3 direction;
};

HitAttribute pack_hit_attribute(ivec3 voxel_i) {
    return HitAttribute(voxel_i.x + voxel_i.y * BLAS_BRICK_SIZE + voxel_i.z * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE);
}

RayPayload pack_ray_payload(uint blas_id, uint brick_id, HitAttribute hit_attrib) {
    return RayPayload(blas_id, (brick_id * (BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * 2)) | hit_attrib.data);
}

RayPayload miss_ray_payload() {
    return pack_ray_payload(0, 0, HitAttribute(BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE));
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
    if (all(greaterThanEqual(r.origin, aabb.minimum)) && all(lessThanEqual(r.origin, aabb.maximum))) {
        return 0.0;
    }
    vec3 invDir = 1.0 / r.direction;
    vec3 tbot = invDir * (aabb.minimum - r.origin);
    vec3 ttop = invDir * (aabb.maximum - r.origin);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    float t0 = max(tmin.x, max(tmin.y, tmin.z));
    float t1 = min(tmax.x, min(tmax.y, tmax.z));
    return t1 > max(t0, 0.0) ? t0 : -1.0;
}

float hitAabb_midpoint(const Aabb aabb, const Ray r) {
    vec3 invDir = 1.0 / r.direction;
    vec3 tbot = invDir * (aabb.minimum - r.origin);
    vec3 ttop = invDir * (aabb.maximum - r.origin);
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

PackedVoxel unpack_ray_payload(
    daxa_BufferPtr(daxa_BufferPtr(BlasGeom)) geometry_pointers,
    daxa_BufferPtr(daxa_BufferPtr(VoxelBrickAttribs)) attribute_pointers,
    daxa_BufferPtr(VoxelBlasTransform) blas_transforms,
    RayPayload payload, Ray ray, out vec3 hit_pos, out vec3 hit_vel) {
    uint blas_id = payload.data0;
    uint brick_id = payload.data1 / (BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * 2);
    uint voxel_index = payload.data1 & (BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * 2 - 1);
    daxa_BufferPtr(VoxelBrickAttribs) brick_attribs = deref(advance(attribute_pointers, blas_id));
    daxa_BufferPtr(BlasGeom) blas_geoms = deref(advance(geometry_pointers, blas_id));
    {
        // mat3x4 m = deref(advance(blas_transforms, blas_id));
        // mat4 world_to_blas = mat4(m[0], m[1], m[2], vec4(0, 0, 0, 1));
        // mat4 blas_to_world = transpose(world_to_blas);

        hit_vel = deref(advance(blas_transforms, blas_id)).vel;

        vec3 v = deref(advance(blas_transforms, blas_id)).pos;
        Aabb aabb = deref(advance(blas_geoms, brick_id)).aabb;
        ivec3 mapPos = ivec3(voxel_index % BLAS_BRICK_SIZE, (voxel_index / BLAS_BRICK_SIZE) % BLAS_BRICK_SIZE, voxel_index / BLAS_BRICK_SIZE / BLAS_BRICK_SIZE);
        aabb.minimum = vec3(ivec3(floor(aabb.minimum * VOXEL_SCL)) & ~0x7) * VOXEL_SIZE;
        aabb.minimum += vec3(mapPos) * VOXEL_SIZE;
        aabb.maximum = aabb.minimum + VOXEL_SIZE;
        ray.origin -= v;
        hit_pos = ray.origin + ray.direction * hitAabb(aabb, ray);
        hit_pos += v;
        // hit_pos = (blas_to_world * vec4(hit_pos, 1)).xyz;
    }
    return deref(advance(brick_attribs, brick_id)).packed_voxels[voxel_index];
}

bool getVoxel(daxa_BufferPtr(BlasGeom) blas_geoms, uint brick_id, ivec3 c) {
    uint bit_index = c.x + c.y * BLAS_BRICK_SIZE + c.z * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE;
    uint u32_index = bit_index / 32;
    uint in_u32_index = bit_index & 0x1f;
    uint val = deref(advance(blas_geoms, brick_id)).bitmask[u32_index];
    return (val & (1 << in_u32_index)) != 0;
}

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_INTERSECTION || DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_CLOSEST_HIT
hitAttributeEXT HitAttribute hit_attrib;
#else
#extension GL_EXT_ray_query : enable
rayQueryEXT ray_query;
HitAttribute hit_attrib;
#endif

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_INTERSECTION || DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_COMPUTE

float traceVoxelDataBitmap(daxa_BufferPtr(BlasGeom) blas_geom, vec3 origin, vec3 dir, out ivec3 coord, inout int nrm, inout int iterCount) {
    float t = 0.0;
    vec3 invDir = vec3(1.0) / (abs(dir) + vec3(0.0001));
    vec3 tSign = sign(dir);
    const vec3 zSign = step(vec3(0.0), tSign);
    const vec3 tDelta = invDir * VOXEL_SIZE;
    const vec3 tPos = clamp(origin * VOXEL_SCL, vec3(0.0), vec3(7.99999)); // <- clamp it to start inside chunk
    const vec3 ti = floor(tPos);
    vec3 tMax = (invDir * (zSign + tSign * (ti - tPos))) * VOXEL_SIZE;
    coord = ivec3(ti);
    const ivec3 coordDelta = ivec3(tSign);
    while (((coord.x | coord.y | coord.z) & 0xFFFFFFF8) == 0) {
        if (getVoxel(blas_geom, 0, coord) == true)
            return t;
        // const int a = deref(chunkPrimitivePtr).bitmap[coord.z * 8 + coord.y];
        // const int mask = 1 << coord.x;
        // if ((mask & a) != 0u)
        //     return t;
        float mi = min(min(tMax.x, tMax.y), tMax.z);
        if (mi == tMax.x) {
            nrm = 0;
            coord.x += coordDelta.x;
            tMax.x += tDelta.x;
        } else if (mi == tMax.y) {
            nrm = 1;
            coord.y += coordDelta.y;
            tMax.y += tDelta.y;
        } else {
            nrm = 2;
            coord.z += coordDelta.z;
            tMax.z += tDelta.z;
        }
        t = mi;
        iterCount++;
    }
    return -1.0;
}

float traceVoxelDataBitmap(daxa_BufferPtr(BlasGeom) blas_geom, vec3 origin, vec3 dir, out ivec3 coord, inout int nrm) {
    int iterCount = 0;
    return traceVoxelDataBitmap(blas_geom, origin, dir, coord, nrm, iterCount);
}

void intersect_voxel_brick(daxa_BufferPtr(daxa_BufferPtr(BlasGeom)) geometry_pointers) {
    Ray ray;

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_INTERSECTION
#define OBJECT_TO_WORLD_MAT gl_ObjectToWorld3x4EXT
#define INSTANCE_CUSTOM_INDEX gl_InstanceCustomIndexEXT
#define PRIMITIVE_INDEX gl_PrimitiveID
    ray.origin = gl_ObjectRayOriginEXT;
    ray.direction = gl_ObjectRayDirectionEXT;
#else
    const mat3x4 object_to_world_mat = transpose(rayQueryGetIntersectionObjectToWorldEXT(ray_query, false));
#define OBJECT_TO_WORLD_MAT object_to_world_mat
#define INSTANCE_CUSTOM_INDEX rayQueryGetIntersectionInstanceCustomIndexEXT(ray_query, false)
#define PRIMITIVE_INDEX rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, false)
    ray.origin = rayQueryGetIntersectionObjectRayOriginEXT(ray_query, false);
    ray.direction = rayQueryGetIntersectionObjectRayDirectionEXT(ray_query, false);
#endif

    ray.origin = (OBJECT_TO_WORLD_MAT * ray.origin).xyz;
    ray.direction = (OBJECT_TO_WORLD_MAT * ray.direction).xyz;

    daxa_BufferPtr(BlasGeom) blas_geoms = deref(advance(geometry_pointers, INSTANCE_CUSTOM_INDEX));
    Aabb aabb = deref(advance(blas_geoms, PRIMITIVE_INDEX)).aabb;
#if 0
    vec2 t = ray_aabb(ray.origin - aabb.minimum, ray.direction, aabb.maximum - aabb.minimum);
    t.x = max(0.0, t.x);

    if (t.x < t.y) {
        vec3 o = ray.origin - vec3(ivec3(floor(aabb.minimum * VOXEL_SCL)) & ~0x7) * VOXEL_SIZE + ray.direction * t.x;

        ivec3 hitCoord;

        int nrm = ray_aabb_normal(ray.origin - aabb.minimum, ray.direction, aabb.maximum - aabb.minimum);
        float hitDist = traceVoxelDataBitmap(advance(blas_geoms, PRIMITIVE_INDEX), o, ray.direction, hitCoord, nrm);
        if (hitDist >= 0.0) {
            float tHit = t.x + hitDist;
#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_INTERSECTION
            hit_attrib = pack_hit_attribute(hitCoord);
            reportIntersectionEXT(tHit, 0);
#else
            if (tHit < rayQueryGetIntersectionTEXT(ray_query, true)) {
                hit_attrib = pack_hit_attribute(hitCoord);
                rayQueryGenerateIntersectionEXT(ray_query, tHit);
            }
#endif
        }
    }
#else
    float tHit = -1;
    tHit = hitAabb(aabb, ray);
    const float BIAS = uintBitsToFloat(0x3f800040); // uintBitsToFloat(0x3f800040) == 1.00000762939453125
    ray.origin += ray.direction * tHit * BIAS;
    if (tHit >= 0) {
        ivec3 bmin = ivec3(floor(aabb.minimum * VOXEL_SCL));
        ivec3 cbmin = bmin & ~0x7;
        ivec3 mapPos = clamp(ivec3(floor(ray.origin * VOXEL_SCL)) - bmin, ivec3(0), ivec3(BLAS_BRICK_SIZE - 1));
        vec3 deltaDist = abs(vec3(length(ray.direction)) / ray.direction);
        vec3 sideDist = (sign(ray.direction) * (vec3(mapPos + bmin) - ray.origin * VOXEL_SCL) + (sign(ray.direction) * 0.5) + 0.5) * deltaDist;
        ivec3 rayStep = ivec3(sign(ray.direction));
        bvec3 mask = lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy));
        for (int i = 0; i < int(3 * VOXEL_SCL); i++) {
            if (getVoxel(blas_geoms, PRIMITIVE_INDEX, mapPos) == true) {
                aabb.minimum += vec3(mapPos) * VOXEL_SIZE;
                aabb.maximum = aabb.minimum + VOXEL_SIZE;
                tHit += hitAabb_midpoint(aabb, ray);
#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_INTERSECTION
                hit_attrib = pack_hit_attribute(mapPos);
                reportIntersectionEXT(tHit, 0);
#else
                if (tHit < rayQueryGetIntersectionTEXT(ray_query, true)) {
                    hit_attrib = pack_hit_attribute(mapPos);
                    rayQueryGenerateIntersectionEXT(ray_query, tHit);
                }
#endif
                break;
            }
            mask = lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy));
            // if (int(mask.x) + int(mask.y) + int(mask.z) > 1) {
            //     if (mask.x && mask.y && mask.z) {
            //         mask.yz = bvec2(false);
            //     } else if (mask.x && mask.y) {
            //         mask.y = false;
            //     } else {
            //         mask.z = false;
            //     }
            // }
            sideDist += vec3(mask) * deltaDist;
            mapPos += ivec3(vec3(mask)) * rayStep;
            bool outside_l = any(lessThan(mapPos, ivec3(0)));
            bool outside_g = any(greaterThanEqual(mapPos, ivec3(BLAS_BRICK_SIZE)));
            if ((int(outside_l) | int(outside_g)) != 0) {
                break;
            }
        }
    }
#endif
}
#endif

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_INTERSECTION
void main() {
    intersect_voxel_brick(push.uses.geometry_pointers);
}
#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_CLOSEST_HIT
layout(location = PAYLOAD_LOC) rayPayloadInEXT RayPayload prd;
void main() {
    prd = pack_ray_payload(gl_InstanceCustomIndexEXT, gl_PrimitiveID, hit_attrib);
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

VoxelTraceResult voxel_trace(in VoxelRtTraceInfo info, in out vec3 ray_pos) {
    VoxelTraceResult result;

    const uint ray_flags = gl_RayFlagsNoOpaqueEXT;
    const uint cull_mask = 0xFF & ~(0x01);
    // const uint cull_mask = 0xFF;
    const uint sbt_record_offset = 0;
    const uint sbt_record_stride = 0;
    const uint miss_index = 0;
    const float t_min = 0.001;
    const float t_max = MAX_DIST;
    HitAttribute nearest_hit_attrib = HitAttribute(0);
    rayQueryInitializeEXT(
        ray_query, accelerationStructureEXT(info.ptrs.tlas),
        ray_flags, cull_mask, ray_pos, t_min, info.ray_dir, t_max);
    while (rayQueryProceedEXT(ray_query)) {
        uint type = rayQueryGetIntersectionTypeEXT(ray_query, false);
        if (type == gl_RayQueryCandidateIntersectionAABBEXT) {
            const float t_aabb = rayQueryGetIntersectionTEXT(ray_query, false);
            if (t_aabb < t_max && t_aabb < rayQueryGetIntersectionTEXT(ray_query, true)) {
                intersect_voxel_brick(info.ptrs.geometry_pointers);
            }
        }
    }
    result.dist = rayQueryGetIntersectionTEXT(ray_query, true);
    if (rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionGeneratedEXT) {
        uint instance_custom_index = rayQueryGetIntersectionInstanceCustomIndexEXT(ray_query, true);
        uint prim_index = rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true);
        RayPayload prd = pack_ray_payload(instance_custom_index, prim_index, hit_attrib);
        result.voxel_data = unpack_ray_payload(info.ptrs.geometry_pointers, info.ptrs.attribute_pointers, info.ptrs.blas_transforms, prd, Ray(ray_pos, info.ray_dir), ray_pos, result.vel);
        Voxel voxel = unpack_voxel(result.voxel_data);
        result.nrm = voxel.normal;
    }
    return result;
}
#endif

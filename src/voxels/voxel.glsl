#ifndef VOXEL_GLSL
#define VOXEL_GLSL

#include "voxel.inl"

struct VoxelHit {
    uint8_t x;
    uint8_t y;
    uint8_t z;
    uint8_t nrm;
};

float traceVoxelDataBitmap(daxa_BufferPtr(BrickPrimitive) brickPrimitivePtr, vec3 origin, vec3 dir, out ivec3 coord, inout int nrm, inout int iterCount) {
    float t = 0.0;
    vec3 invDir = vec3(1.0) / (abs(dir) + vec3(0.0001));
    vec3 tSign = sign(dir);
    const vec3 zSign = step(vec3(0.0), tSign);
    const float voxSize = 1; // deref(brickPrimitivePtr).voxelSize;
    const vec3 tDelta = invDir * voxSize;
    const vec3 tPos = clamp(origin / voxSize, vec3(0.0), vec3(7.99999)); // <- clamp it to start inside brick
    const vec3 ti = floor(tPos);
    vec3 tMax = (invDir * (zSign + tSign * (ti - tPos))) * voxSize;
    coord = ivec3(ti);
    const ivec3 coordDelta = ivec3(tSign);
    while (((coord.x | coord.y | coord.z) & 0xFFFFFFF8) == 0) {
        const int a = deref(brickPrimitivePtr).bitmap[coord.z * 8 + coord.y];
        const int mask = 1 << coord.x;
        if ((mask & a) != 0u)
            return t;
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

float traceVoxelDataBitmap(daxa_BufferPtr(BrickPrimitive) brickPrimitivePtr, vec3 origin, vec3 dir, out ivec3 coord, inout int nrm) {
    int iterCount = 0;
    return traceVoxelDataBitmap(brickPrimitivePtr, origin, dir, coord, nrm, iterCount);
}

#endif

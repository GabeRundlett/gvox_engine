#pragma once

#include <voxels/core.glsl>

struct GbufferDataPacked {
    uvec4 data0;
};

struct GbufferData {
    vec3 albedo;
    vec3 emissive;
    vec3 normal;
    float roughness;
    float metalness;
};

GbufferData unpack(GbufferDataPacked self) {
    GbufferData res;
    Voxel voxel = unpack_voxel(PackedVoxel(self.data0.x));
    res.albedo = voxel.color;
    res.emissive = uint_urgb9e5_to_f32vec3(self.data0.w);
    res.normal = u16_to_nrm(self.data0.y);
    res.roughness = voxel.roughness;
    res.metalness = 0;
    return res;
}

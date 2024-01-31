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
    res.albedo = voxel.color * float(voxel.material_type == 1);
    res.emissive = voxel.color * float(voxel.material_type == 3) * (1000.0 * voxel.roughness + 1);
    res.normal = u16_to_nrm(self.data0.y);
    res.roughness = voxel.material_type == 1 ? voxel.roughness : 1.0;
    res.metalness = float(voxel.material_type == 2);
    return res;
}

#pragma once

#include <voxels/voxels.glsl>

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
    res.emissive = voxel.color * float(voxel.material_type == 3) * (2.0 * voxel.roughness + 0.01);
    res.normal = u16_to_nrm(self.data0.y);
    res.roughness = (voxel.material_type == 1 || voxel.material_type == 2) ? voxel.roughness : 1.0;
    res.metalness = float(voxel.material_type == 2);
    res.albedo = voxel.color * float(voxel.material_type == 1 || voxel.material_type == 2);
    return res;
}

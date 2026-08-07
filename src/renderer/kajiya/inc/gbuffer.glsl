#pragma once

#include <voxels/voxel.glsl>
#include <utilities/gpu/normal.glsl>

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
    GpuVoxel voxel;
    voxel.albedo = vec3(0.5);
    voxel.normal = vec3(0,0,1);
    voxel.roughness = 1;
    voxel.material_type = 1;
    res.emissive = voxel.albedo * float(voxel.material_type == 3) * (voxel.roughness + 0.01);
    res.normal = u16_to_nrm(self.data0.y);
    res.roughness = (voxel.material_type == 1 || voxel.material_type == 2) ? voxel.roughness : 1.0;
    res.metalness = float(voxel.material_type == 2);
    res.albedo = voxel.albedo * float(voxel.material_type == 1 || voxel.material_type == 2);
    return res;
}

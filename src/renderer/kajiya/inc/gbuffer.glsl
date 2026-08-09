#pragma once

#include <utilities/gpu/normal.glsl>
#include <voxels/pack_unpack.inl>

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

GbufferDataPacked pack(GbufferData self) {
    return GbufferDataPacked(uvec4(0, 0, 0, 0));
}

GbufferData unpack(GbufferDataPacked self) {
    GbufferData res;
    Voxel voxel = unpack_voxel(PackedVoxel(self.data0.x));
    res.emissive = voxel.albedo * float(voxel.material_type == 2) * (voxel.roughness + 0.01);
    res.normal = u16_to_nrm(self.data0.y);
    res.roughness = (voxel.material_type == 0 || voxel.material_type == 1) ? voxel.roughness : 1.0;
    res.metalness = float(voxel.material_type == 1);
    res.albedo = voxel.albedo * float(voxel.material_type == 0 || voxel.material_type == 1);
    return res;
}

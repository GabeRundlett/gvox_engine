#pragma once

#include "shared.inl"
#include "common/interface/raytrace.hlsl"

struct Camera {
    float4x4 inv_mat;
    float3x3 rot_mat;
    float3 pos;
    float2 render_size, inv_frame_dim;
    float fov, tan_half_fov, aspect;

    void default_init() {
        inv_mat = float4x4(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        rot_mat = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
        pos = float3(0, 0, 0);
        render_size = float2(0, 0);
        fov = 0;
        tan_half_fov = 0;
        aspect = 0;
    }

    void update(StructuredBuffer<GpuInput> input);

    float2 create_view_uv(float2 pixel_i);
    Ray create_view_ray(float2 uv);
    uint2 world2screen(float3 worldspace_p);
};

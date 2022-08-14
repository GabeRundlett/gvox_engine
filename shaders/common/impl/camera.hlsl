#pragma once

#include "common/impl/raytrace.hlsl"
#include "utils/math.hlsl"

float2 Camera::create_view_uv(float2 pixel_i) {
    float2 result = pixel_i * inv_frame_dim * 2 - 1;
    result.x *= aspect;
    return result;
}

Ray Camera::create_view_ray(float2 uv) {
    Ray result;
    result.o = pos;

#if 0
    {
        float r0 = atan2(uv.y, uv.x);
        float r1 = length(uv) * fov * 3.14159f / 360.0f;
        float sin_r1 = sin(r1);
        float cos_r1 = cos(r1);
        result.nrm = normalize(float3(cos(r0) * sin_r1, cos_r1, -sin(r0) * sin_r1));
    }
#elif 0
    {
        float r0 = -uv.x * fov * 3.14159f / 360.0f;
        float r1 = uv.y * fov * 3.14159f / 360.0f;
        float cos_r0 = cos(r0), sin_r0 = sin(r0);
        float cos_r1 = cos(r1), sin_r1 = sin(r1);

        float3x3 ry = float3x3(
            +cos_r0, 0, -sin_r0,
            0, 1, 0,
            +cos_r0, 0, +cos_r0);
        float3x3 rx = float3x3(
            1, 0, 0,
            0, +cos_r1, +sin_r1,
            0, -sin_r1, +cos_r1);

        float3 r = mul(ry, mul(rx, float3(0, 1, 0)));
        result.nrm = normalize(r);
        // cam_ray.nrm = normalize(front * r.z + right * r.x + up * r.y);
    }
#else
    result.nrm = normalize(float3(uv.x * tan_half_fov, 1, -uv.y * tan_half_fov));
#endif
    result.nrm = mul(rot_mat, result.nrm);
    result.inv_nrm = 1 / result.nrm;
    return result;
}

uint2 Camera::world2screen(float3 worldspace_p) {
    float3 viewspace_p = mul(inv_mat, float4(worldspace_p - pos, 1)).xyz;
    float2 prev_uv = viewspace_p.xy / (viewspace_p.z * tan_half_fov);
    return uint2(round((prev_uv * float2(0.5 / aspect, 0.5) + float2(0.5, 0.5)) * frame_dim));
}

void Camera::update(in out Input input) {
    frame_dim = input.frame_dim;
    inv_frame_dim = 1 / frame_dim;
    aspect = frame_dim.x * inv_frame_dim.y;
    fov = input.fov;
    tan_half_fov = tan(fov * 3.14159f / 360.0f);
    inv_mat = inverse(float4x4(
        rot_mat[0], 0,
        rot_mat[1], 0,
        rot_mat[2], 0,
        pos, 1));
}

#pragma once

// NOTE(grundlett): Merged together frame_constants.hlsl and uv.hlsl

#include <utilities/gpu/math.glsl>
#include <application/globals.inl>

vec2 get_uv(ivec2 pix, vec4 tex_size) { return (vec2(pix) + 0.5) * tex_size.zw; }
vec2 get_uv(vec2 pix, vec4 tex_size) { return (pix + 0.5) * tex_size.zw; }
vec2 cs_to_uv(vec2 cs) { return cs * vec2(0.5, -0.5) + vec2(0.5, 0.5); }
vec2 uv_to_cs(vec2 uv) { return (uv - 0.5) * vec2(2, -2); }

struct ViewRayContext {
    vec4 ray_dir_cs;
    vec4 ray_dir_vs_h;
    vec4 ray_dir_ws_h;
    vec4 ray_origin_cs;
    vec4 ray_origin_vs_h;
    vec4 ray_origin_ws_h;
    vec4 ray_hit_cs;
    vec4 ray_hit_vs_h;
    vec4 ray_hit_ws_h;
};

vec3 biased_ray_origin_ws_modifier(vec3 origin, vec3 normal) {
#if PER_VOXEL_NORMALS
    return floor(origin * VOXEL_SCL) / VOXEL_SCL + normal * 1.5 / VOXEL_SCL;
#else
    return origin;
#endif
}

vec3 ray_dir_vs(in ViewRayContext vrc) { return normalize(vrc.ray_dir_vs_h.xyz); }
vec3 ray_dir_ws(in ViewRayContext vrc) { return normalize(vrc.ray_dir_ws_h.xyz); }
vec3 ray_origin_vs(in ViewRayContext vrc) { return vrc.ray_origin_vs_h.xyz / vrc.ray_origin_vs_h.w; }
vec3 ray_origin_ws(in ViewRayContext vrc) { return vrc.ray_origin_ws_h.xyz / vrc.ray_origin_ws_h.w; }
vec3 ray_hit_vs(in ViewRayContext vrc) { return vrc.ray_hit_vs_h.xyz / vrc.ray_hit_vs_h.w; }
vec3 ray_hit_ws(in ViewRayContext vrc) { return vrc.ray_hit_ws_h.xyz / vrc.ray_hit_ws_h.w; }
vec3 biased_secondary_ray_origin_ws(in ViewRayContext vrc) {
    return ray_hit_ws(vrc) - ray_dir_ws(vrc) * (length(ray_hit_vs(vrc)) + length(ray_hit_ws(vrc))) * 1e-4;
}
vec3 biased_secondary_ray_origin_ws_with_normal(in ViewRayContext vrc, vec3 normal) {
    vec3 ws_abs = abs(ray_hit_ws(vrc));
    float max_comp = max(max(ws_abs.x, ws_abs.y), max(ws_abs.z, -ray_hit_vs(vrc).z));
    vec3 origin = ray_hit_ws(vrc) + (normal - ray_dir_ws(vrc)) * max(1e-4, max_comp * 1e-6);
    return biased_ray_origin_ws_modifier(origin, normal);
}
ViewRayContext vrc_from_uv(daxa_BufferPtr(GpuInput) gpu_input, vec2 uv) {
    ViewRayContext res;
    res.ray_dir_cs = vec4(uv_to_cs(uv), 0.0, 1.0);
    res.ray_dir_vs_h = deref(gpu_input).player.cam.sample_to_view * res.ray_dir_cs;
    res.ray_dir_ws_h = deref(gpu_input).player.cam.view_to_world * res.ray_dir_vs_h;
    res.ray_origin_cs = vec4(uv_to_cs(uv), 1.0, 1.0);
    res.ray_origin_vs_h = deref(gpu_input).player.cam.sample_to_view * res.ray_origin_cs;
    res.ray_origin_ws_h = deref(gpu_input).player.cam.view_to_world * res.ray_origin_vs_h;
    return res;
}
ViewRayContext unjittered_vrc_from_uv(daxa_BufferPtr(GpuInput) gpu_input, vec2 uv) {
    ViewRayContext res;
    res.ray_dir_cs = vec4(uv_to_cs(uv), 0.0, 1.0);
    res.ray_dir_vs_h = deref(gpu_input).player.cam.clip_to_view * res.ray_dir_cs;
    res.ray_dir_ws_h = deref(gpu_input).player.cam.view_to_world * res.ray_dir_vs_h;
    res.ray_origin_cs = vec4(uv_to_cs(uv), 1.0, 1.0);
    res.ray_origin_vs_h = deref(gpu_input).player.cam.clip_to_view * res.ray_origin_cs;
    res.ray_origin_ws_h = deref(gpu_input).player.cam.view_to_world * res.ray_origin_vs_h;
    return res;
}
ViewRayContext vrc_from_uv_and_depth(daxa_BufferPtr(GpuInput) gpu_input, vec2 uv, float depth) {
    ViewRayContext res;
    res.ray_dir_cs = vec4(uv_to_cs(uv), 0.0, 1.0);
    res.ray_dir_vs_h = deref(gpu_input).player.cam.sample_to_view * res.ray_dir_cs;
    res.ray_dir_ws_h = deref(gpu_input).player.cam.view_to_world * res.ray_dir_vs_h;
    res.ray_origin_cs = vec4(uv_to_cs(uv), 1.0, 1.0);
    res.ray_origin_vs_h = deref(gpu_input).player.cam.sample_to_view * res.ray_origin_cs;
    res.ray_origin_ws_h = deref(gpu_input).player.cam.view_to_world * res.ray_origin_vs_h;
    res.ray_hit_cs = vec4(uv_to_cs(uv), depth, 1.0);
    res.ray_hit_vs_h = deref(gpu_input).player.cam.sample_to_view * res.ray_hit_cs;
    res.ray_hit_ws_h = deref(gpu_input).player.cam.view_to_world * res.ray_hit_vs_h;
    return res;
}
#define BIAS uintBitsToFloat(0x3f800040) // uintBitsToFloat(0x3f800040) == 1.00000762939453125
ViewRayContext vrc_from_uv_and_biased_depth(daxa_BufferPtr(GpuInput) gpu_input, vec2 uv, float depth) {
    return vrc_from_uv_and_depth(gpu_input, uv, min(1.0, depth * BIAS));
}

vec3 get_eye_position(daxa_BufferPtr(GpuInput) gpu_input) {
    vec4 eye_pos_h = deref(gpu_input).player.cam.view_to_world * vec4(0, 0, 0, 1);
    return eye_pos_h.xyz / eye_pos_h.w + deref(gpu_input).player.player_unit_offset;
}
vec3 get_prev_eye_position(daxa_BufferPtr(GpuInput) gpu_input) {
    vec4 eye_pos_h = deref(gpu_input).player.cam.prev_view_to_prev_world * vec4(0, 0, 0, 1);
    return eye_pos_h.xyz / eye_pos_h.w + deref(gpu_input).player.player_unit_offset;
}
vec3 direction_view_to_world(daxa_BufferPtr(GpuInput) gpu_input, vec3 v) {
    return (deref(gpu_input).player.cam.view_to_world * vec4(v, 0)).xyz;
}
vec3 direction_world_to_view(daxa_BufferPtr(GpuInput) gpu_input, vec3 v) {
    return (deref(gpu_input).player.cam.world_to_view * vec4(v, 0)).xyz;
}
vec3 position_world_to_view(daxa_BufferPtr(GpuInput) gpu_input, vec3 v) {
    return (deref(gpu_input).player.cam.world_to_view * vec4(v, 1)).xyz;
}
vec3 position_view_to_world(daxa_BufferPtr(GpuInput) gpu_input, vec3 v) {
    return (deref(gpu_input).player.cam.view_to_world * vec4(v, 1)).xyz;
}

vec3 position_world_to_sample(daxa_BufferPtr(GpuInput) gpu_input, vec3 v) {
    vec4 p = deref(gpu_input).player.cam.world_to_view * vec4(v, 1);
    p = deref(gpu_input).player.cam.view_to_sample * p;
    return p.xyz / p.w;
}

vec3 position_world_to_clip(daxa_BufferPtr(GpuInput) gpu_input, vec3 v) {
    vec4 p = (deref(gpu_input).player.cam.world_to_view * vec4(v, 1));
    p = (deref(gpu_input).player.cam.view_to_clip * p);
    return p.xyz / p.w;
}

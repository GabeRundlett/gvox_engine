#pragma once

#include <utils/math.glsl>
#include <shared/globals.inl>

daxa_f32vec2 get_uv(daxa_i32vec2 pix, daxa_f32vec4 tex_size) { return (daxa_f32vec2(pix) + 0.5) * tex_size.zw; }
daxa_f32vec2 get_uv(daxa_f32vec2 pix, daxa_f32vec4 tex_size) { return (pix + 0.5) * tex_size.zw; }
daxa_f32vec2 cs_to_uv(daxa_f32vec2 cs) { return cs * daxa_f32vec2(0.5, -0.5) + daxa_f32vec2(0.5, 0.5); }
daxa_f32vec2 uv_to_cs(daxa_f32vec2 uv) { return (uv - 0.5) * daxa_f32vec2(2, -2); }
daxa_f32vec2 uv_to_ss(daxa_BufferPtr(GpuInput) gpu_input, daxa_f32vec2 uv, daxa_f32vec4 tex_size) { return uv - deref(gpu_input).halton_jitter.xy * tex_size.zw * 1.0; }
daxa_f32vec2 ss_to_uv(daxa_BufferPtr(GpuInput) gpu_input, daxa_f32vec2 ss, daxa_f32vec4 tex_size) { return ss + deref(gpu_input).halton_jitter.xy * tex_size.zw * 1.0; }

struct ViewRayContext {
    daxa_f32vec4 ray_dir_cs;
    daxa_f32vec4 ray_dir_vs_h;
    daxa_f32vec4 ray_dir_ws_h;
    daxa_f32vec4 ray_origin_cs;
    daxa_f32vec4 ray_origin_vs_h;
    daxa_f32vec4 ray_origin_ws_h;
    daxa_f32vec4 ray_hit_cs;
    daxa_f32vec4 ray_hit_vs_h;
    daxa_f32vec4 ray_hit_ws_h;
};

vec3 biased_ray_origin_ws_modifier(vec3 origin, vec3 normal) {
#if PER_VOXEL_NORMALS
    return floor(origin * VOXEL_SCL) / VOXEL_SCL + normal * 1.5 / VOXEL_SCL;
#else
    return origin;
#endif
}

daxa_f32vec3 ray_dir_vs(in ViewRayContext vrc) { return normalize(vrc.ray_dir_vs_h.xyz); }
daxa_f32vec3 ray_dir_ws(in ViewRayContext vrc) { return normalize(vrc.ray_dir_ws_h.xyz); }
daxa_f32vec3 ray_origin_vs(in ViewRayContext vrc) { return vrc.ray_origin_vs_h.xyz / vrc.ray_origin_vs_h.w; }
daxa_f32vec3 ray_origin_ws(in ViewRayContext vrc) { return vrc.ray_origin_ws_h.xyz / vrc.ray_origin_ws_h.w; }
daxa_f32vec3 ray_hit_vs(in ViewRayContext vrc) { return vrc.ray_hit_vs_h.xyz / vrc.ray_hit_vs_h.w; }
daxa_f32vec3 ray_hit_ws(in ViewRayContext vrc) { return vrc.ray_hit_ws_h.xyz / vrc.ray_hit_ws_h.w; }
daxa_f32vec3 biased_secondary_ray_origin_ws(in ViewRayContext vrc) {
    return ray_hit_ws(vrc) - ray_dir_ws(vrc) * (length(ray_hit_vs(vrc)) + length(ray_hit_ws(vrc))) * 1e-4;
}
daxa_f32vec3 biased_secondary_ray_origin_ws_with_normal(in ViewRayContext vrc, daxa_f32vec3 normal) {
    daxa_f32vec3 ws_abs = abs(ray_hit_ws(vrc));
    float max_comp = max(max(ws_abs.x, ws_abs.y), max(ws_abs.z, -ray_hit_vs(vrc).z));
    vec3 origin = ray_hit_ws(vrc) + (normal - ray_dir_ws(vrc)) * max(1e-4, max_comp * 1e-6);
    return biased_ray_origin_ws_modifier(origin, normal);
}
ViewRayContext vrc_from_uv(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec2 uv) {
    ViewRayContext res;
    res.ray_dir_cs = daxa_f32vec4(uv_to_cs(uv), 0.0, 1.0);
    res.ray_dir_vs_h = deref(globals).player.cam.sample_to_view * res.ray_dir_cs;
    res.ray_dir_ws_h = deref(globals).player.cam.view_to_world * res.ray_dir_vs_h;
    res.ray_origin_cs = daxa_f32vec4(uv_to_cs(uv), 1.0, 1.0);
    res.ray_origin_vs_h = deref(globals).player.cam.sample_to_view * res.ray_origin_cs;
    res.ray_origin_ws_h = deref(globals).player.cam.view_to_world * res.ray_origin_vs_h;
    return res;
}
ViewRayContext unjittered_vrc_from_uv(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec2 uv) {
    ViewRayContext res;
    res.ray_dir_cs = daxa_f32vec4(uv_to_cs(uv), 0.0, 1.0);
    res.ray_dir_vs_h = deref(globals).player.cam.clip_to_view * res.ray_dir_cs;
    res.ray_dir_ws_h = deref(globals).player.cam.view_to_world * res.ray_dir_vs_h;
    res.ray_origin_cs = daxa_f32vec4(uv_to_cs(uv), 1.0, 1.0);
    res.ray_origin_vs_h = deref(globals).player.cam.clip_to_view * res.ray_origin_cs;
    res.ray_origin_ws_h = deref(globals).player.cam.view_to_world * res.ray_origin_vs_h;
    return res;
}
ViewRayContext vrc_from_uv_and_depth(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec2 uv, float depth) {
    ViewRayContext res;
    res.ray_dir_cs = daxa_f32vec4(uv_to_cs(uv), 0.0, 1.0);
    res.ray_dir_vs_h = deref(globals).player.cam.sample_to_view * res.ray_dir_cs;
    res.ray_dir_ws_h = deref(globals).player.cam.view_to_world * res.ray_dir_vs_h;
    res.ray_origin_cs = daxa_f32vec4(uv_to_cs(uv), 1.0, 1.0);
    res.ray_origin_vs_h = deref(globals).player.cam.sample_to_view * res.ray_origin_cs;
    res.ray_origin_ws_h = deref(globals).player.cam.view_to_world * res.ray_origin_vs_h;
    res.ray_hit_cs = daxa_f32vec4(uv_to_cs(uv), depth, 1.0);
    res.ray_hit_vs_h = deref(globals).player.cam.sample_to_view * res.ray_hit_cs;
    res.ray_hit_ws_h = deref(globals).player.cam.view_to_world * res.ray_hit_vs_h;
    return res;
}
#define BIAS uintBitsToFloat(0x3f800040) // uintBitsToFloat(0x3f800040) == 1.00000762939453125
ViewRayContext vrc_from_uv_and_biased_depth(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec2 uv, float depth) {
    return vrc_from_uv_and_depth(globals, uv, min(1.0, depth * BIAS));
}

daxa_f32vec3 get_eye_position(daxa_RWBufferPtr(GpuGlobals) globals) {
    daxa_f32vec4 eye_pos_h = deref(globals).player.cam.view_to_world * daxa_f32vec4(0, 0, 0, 1);
    return eye_pos_h.xyz / eye_pos_h.w + deref(globals).player.player_unit_offset;
}
daxa_f32vec3 get_prev_eye_position(daxa_RWBufferPtr(GpuGlobals) globals) {
    daxa_f32vec4 eye_pos_h = deref(globals).player.cam.prev_view_to_prev_world * daxa_f32vec4(0, 0, 0, 1);
    return eye_pos_h.xyz / eye_pos_h.w + deref(globals).player.player_unit_offset;
}
daxa_f32vec3 direction_view_to_world(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec3 v) {
    return (deref(globals).player.cam.view_to_world * daxa_f32vec4(v, 0)).xyz;
}
daxa_f32vec3 direction_world_to_view(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec3 v) {
    return (deref(globals).player.cam.world_to_view * daxa_f32vec4(v, 0)).xyz;
}
daxa_f32vec3 position_world_to_view(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec3 v) {
    return (deref(globals).player.cam.world_to_view * daxa_f32vec4(v, 1)).xyz;
}
daxa_f32vec3 position_view_to_world(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec3 v) {
    return (deref(globals).player.cam.view_to_world * daxa_f32vec4(v, 1)).xyz;
}

daxa_f32vec3 position_world_to_sample(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec3 v) {
    daxa_f32vec4 p = deref(globals).player.cam.world_to_view * daxa_f32vec4(v, 1);
    p = deref(globals).player.cam.view_to_sample * p;
    return p.xyz / p.w;
}

daxa_f32vec3 position_world_to_clip(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec3 v) {
    daxa_f32vec4 p = (deref(globals).player.cam.world_to_view * daxa_f32vec4(v, 1));
    p = (deref(globals).player.cam.view_to_clip * p);
    return p.xyz / p.w;
}

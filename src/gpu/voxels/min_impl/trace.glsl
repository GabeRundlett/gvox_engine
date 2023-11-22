#pragma once

#include <shared/app.inl>

#include <utils/math.glsl>

vec2 opU(vec2 d1, vec2 d2) {
    return (d1.x < d2.x) ? d1 : d2;
}

const float floor_z = -1.0;

vec2 map(in vec3 pos) {
    vec2 res = vec2(pos.z - floor_z, 0.0);
    res = opU(res, vec2(sd_sphere(pos - vec3(+1.0, +1.0, 0.25 + floor_z), 0.25), 1.0));
    res = opU(res, vec2(sd_ellipsoid(pos - vec3(+1.0, +2.0, 0.25 + floor_z), daxa_f32vec3(0.1, 0.2, 0.25)), 1.0));
    res = opU(res, vec2(sd_box(pos - vec3(+2.0, +1.0, 0.25 + floor_z), daxa_f32vec3(0.1, 0.2, 0.25)), 1.0));
    res = opU(res, vec2(sd_box_frame(pos - vec3(+3.0, +1.0, 0.25 + floor_z), daxa_f32vec3(0.1, 0.2, 0.25), 0.02), 1.0));
    res = opU(res, vec2(sd_cylinder(pos - vec3(+2.0, +2.0, 0.25 + floor_z), 0.25, 0.25), 1.0));
    res = opU(res, vec2(sd_torus(pos.yzx - vec3(+3.0, 0.25 + floor_z, +1.0), daxa_f32vec2(0.18, 0.07)), 1.0));
    return res;
}

vec3 map_col(vec3 pos, int id) {
    switch (id) {
    case 0:
        return daxa_f32vec3(0.2) + float(int(floor(pos.x) + floor(pos.y)) & 1) * 0.05;
    case 1:
        return daxa_f32vec3(0.6, 0.04, 0.12) + float(int(floor(pos.x * 8.0) + floor(pos.y * 8.0) + floor(pos.z * 8.0)) & 1) * 0.05;
    default:
        return daxa_f32vec3(0.0);
    }
}

vec3 map_nrm(in vec3 pos) {
    vec3 n = vec3(0.0);
    for (int i = 0; i < 4; i++) {
        vec3 e = 0.5773 * (2.0 * vec3((((i + 3) >> 1) & 1), ((i >> 1) & 1), (i & 1)) - 1.0);
        n += e * map(pos + 0.0005 * e).x;
    }
    return normalize(n);
}

VoxelTraceResult voxel_trace(in VoxelTraceInfo info, in out daxa_f32vec3 ray_pos) {
    VoxelTraceResult result;

    result.dist = 0;
    result.nrm = -info.ray_dir;
    result.step_n = 0;
    result.voxel_data = 0;
    result.vel = daxa_f32vec3(deref(info.ptrs.globals).offset - deref(info.ptrs.globals).prev_offset);

    daxa_f32vec3 offset = daxa_f32vec3(deref(info.ptrs.globals).offset);
    ray_pos += offset;

#if TRACE_DEPTH_PREPASS_COMPUTE
    ray_pos -= offset;
    return result;
#endif

    if (info.extend_to_max_dist) {
        result.dist = MAX_DIST;
    }

    ray_pos += info.ray_dir * 0.001 / VOXEL_SCL;

    vec3 ro = ray_pos;
    vec3 rd = info.ray_dir;

    vec2 res = vec2(-1.0, -1.0);

    float tmin = 0.0;
    float tmax = info.max_dist;

    // raytrace floor plane
    float tp1 = (floor_z - ro.z) / rd.z;
    if (tp1 > 0.0) {
        tmax = min(tmax, tp1);
        res = vec2(tp1, 0.0);
    }

    float t = tmin;
    int i = 0;
    for (; i < info.max_steps && t < tmax; i++) {
        vec2 h = map(ro + rd * t);
        if (abs(h.x) < (0.0001 * t)) {
            res = vec2(t, h.y);
            break;
        }
        t += h.x;
    }

    if (res.y != -1.0) {
        result.dist = res.x;
        ray_pos += rd * result.dist;

        result.nrm = map_nrm(ro + rd * t);
        daxa_f32vec3 col = map_col(ray_pos, int(res.y));
        daxa_u32 id = 1;

        result.voxel_data = daxa_f32vec4_to_uint_rgba8(daxa_f32vec4(col, 0.0)) | (id << 0x18);
    }

    ray_pos -= offset;
    return result;
}
